# ======================================================================
# train_jeu1_plateau.py
# Rôle     : [Copie de train_jeu1.py avec --plateau, 23 nov] Court entraînement du réseau d'étoiles, depuis zéro, sur le jeu 1 (paires du retrait classique, 22 nov).
# Étape    : 6. Entraînement sur paires réelles (recette de base, 22 nov) (voir scripts/LISEZMOI.md)
# Entrées  : paires réelles .npz (input, target, weight), départ E3
# Sorties  : instantanés .pt, history.json
# Exemple  : python train_jeu1.py --tag J1 --hours 2 --device cuda:0 --tenues rosette_decoupe6,m57_a_m57
# ======================================================================
"""[Copie de train_jeu1.py avec --plateau, 23 nov] Court entraînement du réseau d'étoiles, depuis zéro, sur le jeu 1 (paires du retrait classique, 22 nov).

Recette de la chaîne (train_e.py, gardée telle quelle : NAFNet de la chaîne, tuiles 256, L1 sur l'étirement asinh fixe,
AdamW 5e-4 (0,9 ; 0,9), décroissance 1e-4, 1000 pas de montée puis cosinus, gradient coupé à 0,1, lots de 8, bf16, 60 %
des tuiles avec étoiles tirées à la volée), deux changements seulement :
- les exemples : les paires du jeu 1 (<DONNEES>\\Paires étoiles jeu 1 – 22 nov\\*.npz : input, target, weight) ;
- la perte : L1 pondérée par la carte de poids de la paire (0 = mal inventé, 0,5 = réparation bonne, 1 = vrais pixels et
  inventions inévitables bonnes), divisée par la somme des poids.
Validation : les paires passées à --tenues (jamais vues à l'entraînement), toutes les 1000 pas.

    python train_jeu1.py --tag J1 --hours 2 --device cuda:0 --tenues rosette_decoupe6,m57_a_m57
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from models import chain_nafnet, fwd  # noqa: E402
from train_e import synthetic_stars  # noqa: E402
sys.path.insert(0, str(RACINE / "Etoiles synthetiques – 22 nov"))

PAIRES = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles jeu 1 – 22 nov")
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles jeu 1 – 22 nov")
PATCH = 256


class Paires:
    """Les paires en mémoire ; tirage de tuiles 256 au hasard (rotations, miroirs), une tuile à moins de 30 % de poids
    est rejetée."""

    def __init__(self, fichiers, seed):
        self.items = []
        for f in fichiers:
            z = np.load(f)
            I, T, W = z["input"].astype(np.float32), z["target"].astype(np.float32), z["weight"].astype(np.float32)
            lim = I.reshape(3, -1).max(1)[:, None, None]
            fw = float(z["fwhm"]) if "fwhm" in z.files else 3.0
            self.items.append((I, T, W, lim, fw, f.stem))
        self.rng = np.random.default_rng(seed)

    def get(self, n):
        xin, tgt, wts, fws, top, lims = [], [], [], [], [], []
        while len(xin) < n:
            I, T, W, lim, fw, _ = self.items[int(self.rng.integers(len(self.items)))]
            H, Wd = I.shape[1:]
            y, x = int(self.rng.integers(0, H - PATCH + 1)), int(self.rng.integers(0, Wd - PATCH + 1))
            a = I[:, y:y + PATCH, x:x + PATCH]
            t = np.minimum(T[:, y:y + PATCH, x:x + PATCH], lim)
            w = W[None, y:y + PATCH, x:x + PATCH]
            if w.mean() < 0.3:
                continue
            k = int(self.rng.integers(4))
            a, t, w = np.rot90(a, k, (1, 2)), np.rot90(t, k, (1, 2)), np.rot90(w, k, (1, 2))
            if self.rng.random() < 0.5:
                a, t, w = a[:, :, ::-1], t[:, :, ::-1], w[:, :, ::-1]
            xin.append(np.ascontiguousarray(a)); tgt.append(np.ascontiguousarray(t)); wts.append(np.ascontiguousarray(w))
            fws.append(fw); top.append(float(lim.max())); lims.append(lim.ravel())
        return (np.stack(xin), np.stack(tgt), np.stack(wts), np.array(fws), np.array(top),
                np.stack(lims).astype(np.float32))


class LotsSynth(torch.utils.data.IterableDataset):
    """Lots préparés sur le processeur (22 nov soir) : tuiles des paires réelles, et sur une part ``synthetic`` des tuiles
    les étoiles synthétiques v2 (Etoiles synthetiques – 22 nov\\paires_synthetiques.py, réalisme jugé à l'aveugle) :
    fond = la cible sans étoiles (une fois sur deux) ou l'entrée avec ses vraies étoiles ; entrée = fond + étoiles tirées
    (+ leur bruit de photon) ; cible = la cible de la paire ; poids = celui de la paire (les zones mal inventées restent
    à 0, une étoile tirée dessus n'est pas apprise)."""

    def __init__(self, fichiers, batch, synthetic, seed):
        self.fichiers, self.batch, self.synthetic, self.seed = fichiers, batch, synthetic, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        torch.set_num_threads(2)
        from paires_synthetiques import paire_synthetique
        D = Paires(self.fichiers, self.seed + 1000 * wid)
        rng = np.random.default_rng(self.seed + 7 + 1000 * wid)
        k = 0
        while True:
            x, t, w, _, _, _ = D.get(self.batch)
            for i in range(self.batch):
                if rng.random() < self.synthetic:
                    fond = t[i] if rng.random() < 0.5 else x[i]
                    e, _ = paire_synthetique(np.ascontiguousarray(fond, np.float32), int(rng.integers(2**31)) + k)
                    x[i] = e.astype(np.float32)
                k += 1
            yield torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(w)


def valider(net, V, dev):
    """Erreur pondérée sur les paires tenues, en tuiles 256 jointives ; « sous les étoiles » = là où entrée et cible
    diffèrent nettement (dans l'étirement)."""
    net.eval()
    tot = wsum = under = usum = 0.0
    for I, T, W, lim, _, _ in V.items:
        H, Wd = I.shape[1:]
        for y in range(0, H - PATCH + 1, PATCH):
            for x in range(0, Wd - PATCH + 1, PATCH):
                a = torch.from_numpy(I[None, :, y:y + PATCH, x:x + PATCH].copy()).to(dev)
                t = torch.from_numpy(np.minimum(T[:, y:y + PATCH, x:x + PATCH], lim)[None].copy()).to(dev)
                w = torch.from_numpy(W[None, None, y:y + PATCH, x:x + PATCH].copy()).to(dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    o = net(fwd(a)).float()
                e = (o - fwd(t)).abs().mean(1, keepdim=True)
                st = ((fwd(a) - fwd(t)).mean(1, keepdim=True) > 0.05).float() * w
                tot += float((e * w).sum()); wsum += float(w.sum())
                under += float((e * st).sum()); usum += float(st.sum())
    net.train()
    return {"val": tot / max(wsum, 1.0), "val_etoiles": under / max(usum, 1.0)}
valider = torch.no_grad()(valider)


def plateau(n, h, w, gen, dev):
    """Piédestaux lisses (23 nov) : 1 à 3 taches gaussiennes larges (sigma 80-400 px, centre pouvant sortir de la tuile) d'amplitude
    log-uniforme 0,5-50 unités de bruit, couleur commune à ±15 %, et dans 30 % des cas une pente douce (±60 % du niveau à travers
    la tuile). Ajoutés À L'IDENTIQUE à l'entrée et à la cible : le réseau apprend qu'un fond lisse de quelques unités reste."""
    yy = torch.arange(h, device=dev, dtype=torch.float32)[None, :, None]
    xx = torch.arange(w, device=dev, dtype=torch.float32)[None, None, :]
    out = torch.zeros(n, 3, h, w, device=dev)
    r = lambda *s: torch.rand(*s, device=dev, generator=gen)
    for _ in range(3):
        on = (r(n) < (0.7 if _ else 1.0)).float()
        sg = 80 + 320 * r(n, 1, 1)
        cy, cx = (r(n, 1, 1) * 1.6 - 0.3) * h, (r(n, 1, 1) * 1.6 - 0.3) * w
        amp = torch.exp(np.log(0.5) + r(n) * (np.log(50) - np.log(0.5)))
        col = 0.85 + 0.3 * r(n, 3)
        g = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sg ** 2))
        out += (on * amp)[:, None, None, None] * col[:, :, None, None] * g[:, None]
    pente = (r(n) < 0.3).float()
    ang = r(n, 1, 1) * 2 * np.pi
    ramp = ((xx / w - 0.5) * torch.cos(ang) + (yy / h - 0.5) * torch.sin(ang))
    lvl = out.mean((1, 2, 3))
    out = out + (pente * lvl * 1.2)[:, None, None, None] * ramp[:, None]
    return out.clamp_min(0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="J1")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--synthetic", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--tenues", default="", help="paires tenues à l'écart (noms séparés par des virgules)")
    ap.add_argument("--etoiles", default="v2", choices=["v2", "chaine"],
                    help="v2 = étoiles synthétiques réalistes (processeur, 2 préparateurs) ; chaine = celles de train_e")
    ap.add_argument("--preparateurs", type=int, default=2)
    ap.add_argument("--depart", default="", help="poids de départ (ex. E3 pas 20000, entièrement à nous) ; vide = zéro")
    ap.add_argument("--plateau", type=float, default=0.0, help="part des tuiles (réelles ou synthétiques) recevant un piédestal lisse identique entrée/cible")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    if dev.type == "cuda":
        assert "4060" in torch.cuda.get_device_name(dev), "carte 0 = 4060 Ti seulement (jamais la 3070)"
    torch.backends.cudnn.benchmark = True
    out = SORTIE / a.tag
    (out / "snap").mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(m):
        line = f"{datetime.now():%H:%M:%S} {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    tenues = {s for s in a.tenues.split(",") if s}
    fich = sorted(PAIRES.glob("*.npz"))
    tr = [f for f in fich if f.stem not in tenues]
    va = [f for f in fich if f.stem in tenues]
    cfg = {**vars(a), "paires_entrainement": [f.stem for f in tr], "paires_tenues": [f.stem for f in va],
           "recette": "train_e.py (chaîne) ; exemples = jeu 1 ; perte L1 pondérée"}
    (out / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    D = Paires(tr, 0)
    V = Paires(va, 1) if va else None
    gen = torch.Generator(device=dev); gen.manual_seed(0); torch.manual_seed(0)
    net = chain_nafnet()
    if a.depart:
        net.load_state_dict(torch.load(a.depart, map_location="cpu", weights_only=False)["net"], strict=True)
    net = net.to(dev)
    log(f"{a.tag} : NAFNet de la chaîne, {'départ ' + a.depart if a.depart else 'poids au hasard'},{len(tr)} paires d'entraînement, {len(va)} tenues, "
        f"étoiles tirées sur {a.synthetic:.0%} des tuiles, lots {a.batch}, lr {a.lr}, {a.hours} h")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.9))
    hist, best = [], None
    lots = None
    if a.etoiles == "v2":
        lots = iter(torch.utils.data.DataLoader(LotsSynth(tr, a.batch, a.synthetic, 0), batch_size=None,
                                                num_workers=a.preparateurs, persistent_workers=a.preparateurs > 0,
                                                prefetch_factor=4 if a.preparateurs else None))
    t0, it, run, warm = time.time(), 0, None, 1000
    while (time.time() - t0) < a.hours * 3600:
        frac = (time.time() - t0) / (a.hours * 3600)
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (it + 1) / warm) * 0.5 * (1 + np.cos(np.pi * frac))
        if lots is not None:
            x, t, w = (v.to(dev, non_blocking=True) for v in next(lots))
            pick = torch.zeros(1, dtype=torch.bool)
        else:
            x, t, w, fw, top, lims = D.get(a.batch)
            x, t, w = torch.from_numpy(x).to(dev), torch.from_numpy(t).to(dev), torch.from_numpy(w).to(dev)
            pick = torch.rand(a.batch, generator=gen, device=dev) < a.synthetic
        if pick.any():
            k = int(pick.sum()); sel = pick.cpu().numpy()
            stars = synthetic_stars(k, fw[sel], top[sel], gen, dev)
            base = torch.where(torch.rand(k, 1, 1, 1, generator=gen, device=dev) < 0.5, t[pick], x[pick])
            sat = torch.from_numpy(lims[sel]).to(dev)[:, :, None, None] * (0.5 + 0.5 * torch.rand(k, 1, 1, 1, generator=gen, device=dev))
            x[pick] = torch.minimum(base + stars, torch.maximum(base, sat))
        if a.plateau > 0:
            pl = plateau(x.shape[0], x.shape[2], x.shape[3], gen, dev)
            keep = (torch.rand(x.shape[0], 1, 1, 1, generator=gen, device=dev) < a.plateau).float()
            x = x + pl * keep; t = t + pl * keep
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pred = net(fwd(x)).float()
        loss = ((pred - fwd(t)).abs() * w).sum() / (w.sum() * 3 + 1e-6)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        opt.step()
        run = loss.item() if run is None else 0.99 * run + 0.01 * loss.item()
        it += 1
        if it % 1000 == 0:
            v = valider(net, V, dev) if V else {"val": float("nan"), "val_etoiles": float("nan")}
            hist.append({"pas": it, "minutes": round((time.time() - t0) / 60, 1), "train": round(run, 5),
                         **{k_: round(v_, 5) for k_, v_ in v.items()}})
            (out / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
            ck = {"model": "chain_nafnet", "net": net.state_dict(), "config": cfg, "step": it, "val": v}
            torch.save(ck, out / "last.pt"); torch.save(ck, out / "snap" / f"pas_{it}.pt")
            if V and (best is None or v["val_etoiles"] < best):
                best = v["val_etoiles"]; torch.save(ck, out / "best.pt")
            log(f"pas {it} : train {run:.5f}, validation {v['val']:.5f}, sous les étoiles {v['val_etoiles']:.5f}")
    torch.save({"model": "chain_nafnet", "net": net.state_dict(), "config": cfg, "step": it}, out / "last.pt")
    log(f"fini : {it} pas en {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
