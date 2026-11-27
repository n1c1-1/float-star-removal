# ======================================================================
# train_n3.py
# Role     : COPY of 25 Nov (trials N3a, N3b) of train_nuit.py (24 Nov, unchanged). Additions, nothing else:
# Step     : 10. Discarded trial N3 (large synthetic halos, 25 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33d/v33e pairs
# Outputs  : snapshots, table
# Example  : python train_n3.py [arguments]  (see the docstring; paths via the variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""COPY of 25 Nov (trials N3a, N3b) of train_nuit.py (24 Nov, unchanged). Additions, nothing else:
  --suffixe v33d|v33e: pairs read (N3a v33d, N3b v33e); --part-grands: share of the batches (2 tiles of 512 px, same area as 8 x 256)
  of synthetic tiles with LARGE HALOS at the scale of M27 (imposed stars 3e4-3e5 noise units, halo 20-60 noise units at 60 px, 1 noise unit at
  150-300 px, 384 px window, spikes; background = the clean target of a 512 pair, position on nebula where the weight is 1 over
  100 px); cosine learning rate over max(step/--pas, time/--hours) to finish on time (--hours = target duration).
Night trials of 24 Nov (N1, N2): V33C (train_v33c.py, itself A) with the v33d pairs and the APOD web pairs.

Imports train_jeu1.py WITHOUT modifying it (chain_nafnet network, weighted L1 loss, validate). Start E3 step 20000, batches of 8, lr 5e-4,
1000-step warm-up then cosine over the steps, 60 % synthetic star tiles. No artificial noise added.

N1 = V33C with the new data, nothing else:
  - real pairs <DONNEES>\\Paires étoiles v33 – 23 nov\\*_v33d.npz (denoised and raw), same 6 held-out tiles as V33C;
  - web pairs <DONNEES>\\Paires étoiles APOD – 24 nov\\*_v33d.npz (denoised only);
  - draw: 50 % real denoised, 25 % raw, 25 % APOD (if no APOD pair: the refusal is explicit, see --sans-apod).
N2 = N1 plus two changes (--n2):
  (a) the weights between 0 and 1 (0.5: repair judged good) become 1 in all training pairs; the validation
      along the way keeps the original weights (comparable to N1);
  (b) on --part-extra (0.5) of the synthetic tiles, one or two imposed very bright stars (peak 300 to 3e5 noise units, with
      the generator's big halo and spikes), placed on a real NEBULA background: a place where the target has weight 1 over a
      40 px disc (90 %) and where its smoothed level exceeds 1 noise unit (otherwise no imposed star).
Resume: --reprendre rereads last.pt (network, optimiser, step, elapsed time).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "Entrainement jeu 1 – 22 nov"))
sys.path.insert(0, str(RACINE / "Etoiles synthetiques – 22 nov"))
sys.path.insert(0, str(ICI))
from models import chain_nafnet, fwd  # noqa: E402
import train_jeu1 as T1  # noqa: E402

PAIRES = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles v33 – 23 nov")
APOD = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles APOD – 24 nov")
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N3 – 25 nov")
MOT = re.compile(r"(.+_g\d+_\d+)_(debruitee|brute)_(v33d|v33e)$")
TENUES = ("m13_g1408_1408,m57_g1664_2688,ngc6888_g896_1408,ngc7000_tiers1_g2432_1920,"
          "rosette_40da_g1920_1920,wizard_a7s_g2944_1664")


def lister(tenues, apod_dir=APOD, suf="v33d"):
    deb, bru, ten = [], [], []
    for f in sorted(PAIRES.glob(f"*_{suf}.npz")):
        m = MOT.match(f.stem)
        assert m, f.name
        if m.group(1) in tenues:
            if m.group(2) == "debruitee":
                ten.append(f)
        elif m.group(2) == "debruitee":
            deb.append(f)
        else:
            bru.append(f)
    apod = [f for f in sorted(Path(apod_dir).glob(f"*_{suf}.npz")) if "_brute" not in f.stem]
    return deb, bru, apod, ten


class PairesW(T1.Paires):
    """T1.Paires dont les poids intermédiaires (0 < w < 1) passent à 1 si demi_a_un."""

    def __init__(self, fichiers, seed, demi_a_un=False):
        super().__init__(fichiers, seed)
        if demi_a_un:
            self.items = [(I, T, np.where((W > 0) & (W < 1), 1.0, W).astype(np.float32), lim, fw, st)
                          for I, T, W, lim, fw, st in self.items]


def extras_nebuleuse(t, w, rng, n_max=2):
    """Étoiles très brillantes à poser : [(x, y, pic)] sur fond de nébuleuse réel (poids 1 sur 40 px à 90 %, niveau lissé > 1 bruit)."""
    H, Wd = w.shape[-2:]
    lum = t.mean(0).astype(np.float32)
    niv = cv2.blur(lum, (41, 41))
    wm = cv2.blur((w >= 1.0).astype(np.float32).reshape(H, Wd), (81, 81))
    ok = np.argwhere((niv > 1.0) & (wm > 0.9))
    ok = ok[(ok[:, 0] >= 20) & (ok[:, 0] < H - 20) & (ok[:, 1] >= 20) & (ok[:, 1] < Wd - 20)]
    if len(ok) == 0:
        return []
    out = []
    for _ in range(int(rng.integers(1, n_max + 1))):
        yy, xx = ok[int(rng.integers(len(ok)))]
        out.append((float(xx) + rng.random(), float(yy) + rng.random(), float(np.exp(rng.uniform(np.log(300), np.log(3e5))))))
    return out


def extras_grands(t, w, rng):
    """N3 : 1 ou 2 etoiles a grand halo sur une tuile 512 : [(x, y, pic, A60, gamma, r1)]. Position dans [100, 412[ ou le poids est 1
    sur au moins 95 % de la boite de 201 px (le halo y est appris a poids 1) et ou le niveau lisse de la cible depasse 1 bruit
    (fond de nebuleuse reel). Pic log-uniforme 3e4-3e5 ; A60 log-uniforme 20-60 ; r1 (1 bruit) log-uniforme 150-300 ; gamma = ln A60 / ln(r1/60)."""
    H, Wd = w.shape[-2:]
    lum = t.mean(0).astype(np.float32)
    niv = cv2.blur(lum, (41, 41))
    wm = cv2.blur((w >= 1.0).astype(np.float32).reshape(H, Wd), (201, 201))
    ok = np.argwhere((niv > 1.0) & (wm > 0.95))
    ok = ok[(ok[:, 0] >= 100) & (ok[:, 0] < H - 100) & (ok[:, 1] >= 100) & (ok[:, 1] < Wd - 100)]
    if len(ok) == 0:
        return []
    out = []
    for _ in range(int(rng.integers(1, 3))):
        yy, xx = ok[int(rng.integers(len(ok)))]
        A60 = float(np.exp(rng.uniform(np.log(20), np.log(60)))); r1 = float(np.exp(rng.uniform(np.log(150), np.log(300))))
        out.append((float(xx) + rng.random(), float(yy) + rng.random(), float(np.exp(rng.uniform(np.log(3e4), np.log(3e5)))),
                    A60, float(np.log(A60) / np.log(r1 / 60.0)), r1))
    return out


class Melange:
    """Tuiles tirées dans les lots (nom, probabilité, PairesW), une tuile à la fois."""

    def __init__(self, lots, demi_a_un, seed):
        self.P = [(nom, p, PairesW(f, seed + 10 * i, demi_a_un)) for i, (nom, p, f) in enumerate(lots) if f]
        s = sum(p for _, p, _ in self.P)
        self.p = np.array([p / s for _, p, _ in self.P])
        self.rng = np.random.default_rng(seed + 2)

    def get(self, n):
        parts = [self.P[int(self.rng.choice(len(self.P), p=self.p))][2].get(1) for _ in range(n)]
        return tuple(np.concatenate([p[j] for p in parts]) for j in range(6))

    def get_grand(self, rng):
        """Une tuile 512 entiere (rotation, miroir) d'une paire 512 : fond = la cible (bornee a l'entree), poids de la paire ;
        None si aucune position convenable apres 40 essais."""
        for _ in range(40):
            P = self.P[int(self.rng.choice(len(self.P), p=self.p))][2]
            I, T, W, lim, fw, _ = P.items[int(P.rng.integers(len(P.items)))]
            if I.shape[1] < 512 or I.shape[2] < 512:
                continue
            t = np.minimum(T[:, :512, :512], lim)
            w = W[None, :512, :512]
            ex = extras_grands(t, w, rng)
            if not ex:
                continue
            k = int(rng.integers(4))
            t, w = np.rot90(t, k, (1, 2)), np.rot90(w, k, (1, 2))
            if rng.random() < 0.5:
                t, w = t[:, :, ::-1], w[:, :, ::-1]
            return np.ascontiguousarray(t), np.ascontiguousarray(w), ex
        return None


class Lots(torch.utils.data.IterableDataset):
    def __init__(self, lots, batch, synthetic, seed, n2, part_extra, part_grands=0.0):
        self.lots, self.batch, self.synthetic, self.seed = lots, batch, synthetic, seed
        self.n2, self.part_extra, self.part_grands = n2, part_extra, part_grands

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        torch.set_num_threads(1)
        from paires_synthetiques_n3 import paire_synthetique
        D = Melange(self.lots, self.n2, self.seed + 1000 * wid)
        rng = np.random.default_rng(self.seed + 7 + 1000 * wid)
        k = 0
        while True:
            if self.part_grands and rng.random() < self.part_grands:
                tl, wl, el = [], [], []
                while len(tl) < 2:
                    g = D.get_grand(rng)
                    if g is None:
                        break
                    tl.append(g[0]); wl.append(g[1]); el.append(g[2])
                if len(tl) == 2:
                    xs = []
                    for i in range(2):
                        e, _ = paire_synthetique(tl[i], int(rng.integers(2**31)) + k, extras=el[i])
                        xs.append(e.astype(np.float32)); k += 1
                    yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(tl)), torch.from_numpy(np.stack(wl))
                    continue
            x, t, w, _, _, _ = D.get(self.batch)
            for i in range(self.batch):
                if rng.random() < self.synthetic:
                    fond = t[i] if rng.random() < 0.5 else x[i]
                    ex = extras_nebuleuse(t[i], w[i], rng) if (self.n2 and rng.random() < self.part_extra) else None
                    e, _ = paire_synthetique(np.ascontiguousarray(fond, np.float32), int(rng.integers(2**31)) + k, extras=ex)
                    x[i] = e.astype(np.float32)
                k += 1
            yield torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(w)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n2", action="store_true", help="(a) poids 0,5 -> 1 et (b) étoiles très brillantes sur fonds de nébuleuse")
    ap.add_argument("--suffixe", default="v33d", choices=["v33d", "v33e"])
    ap.add_argument("--part-grands", type=float, default=0.0, help="part des lots en 2 tuiles 512 a grands halos")
    ap.add_argument("--warm", type=int, default=1000)
    ap.add_argument("--part-extra", type=float, default=0.5)
    ap.add_argument("--pas", type=int, default=34500, help="5 h à la vitesse de V33C (116 pas/min)")
    ap.add_argument("--hours", type=float, default=5.3, help="plafond de sécurité")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--synthetic", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--part-brute", type=float, default=0.25)
    ap.add_argument("--part-apod", type=float, default=0.25)
    ap.add_argument("--sans-apod", action="store_true", help="autorise un tirage sans paires web")
    ap.add_argument("--apod-dir", default=str(APOD))
    ap.add_argument("--tenues", default=TENUES)
    ap.add_argument("--preparateurs", type=int, default=2)
    ap.add_argument("--limite", type=int, default=0, help="essai à blanc : n paires de chaque sorte")
    ap.add_argument("--depart", default=r"<ESSAIS>\starRemoval\train\E3\snap\step_20000.pt")
    ap.add_argument("--sortie", default=str(SORTIE))
    ap.add_argument("--reprendre", action="store_true")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    if dev.type == "cuda":
        assert "4060" in torch.cuda.get_device_name(dev), "carte 0 = 4060 Ti seulement (jamais la 3070)"
    torch.backends.cudnn.benchmark = True
    out = Path(a.sortie) / a.tag
    (out / "snap").mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(m):
        line = f"{datetime.now():%H:%M:%S} {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    deb, bru, apod, ten = lister({s for s in a.tenues.split(",") if s}, a.apod_dir, a.suffixe)
    if not apod and not a.sans_apod:
        sys.exit("aucune paire web *_" + a.suffixe + ".npz dans " + a.apod_dir + " (--sans-apod pour s'en passer)")
    if a.limite:
        deb, bru, apod = deb[:a.limite], bru[:a.limite], apod[:a.limite]
    p_deb = 1.0 - a.part_brute - (a.part_apod if apod else 0.0)
    lots = [("debruitees", p_deb, deb), ("brutes", a.part_brute, bru), ("apod", a.part_apod, apod)]
    cfg = {**vars(a), "n_debruitees": len(deb), "n_brutes": len(bru), "n_apod": len(apod), "n_tenues": len(ten),
           "debruitees": [f.stem for f in deb], "brutes": [f.stem for f in bru], "apod": [f.stem for f in apod],
           "tenues_fichiers": [f.stem for f in ten],
           "recette": "train_n3.py ; exemples " + a.suffixe + " + APOD ; grands halos " + str(a.part_grands) + " ; perte L1 pondérée ; aucun bruit ajouté"}
    (out / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    V = T1.Paires(ten, 1) if ten else None
    torch.manual_seed(0)
    net = chain_nafnet()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.9))
    it, ecoule, hist, best = 0, 0.0, [], None
    if a.reprendre and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(ck["net"], strict=True)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        it, ecoule = int(ck["step"]), float(ck.get("elapsed", 0.0))
        hist = json.loads((out / "history.json").read_text(encoding="utf-8")) if (out / "history.json").exists() else []
        hist = [h for h in hist if h["pas"] <= it]
        best = min([h["val_etoiles"] for h in hist], default=None)
        net = net.to("cpu")
        log(f"{a.tag} : REPRISE au pas {it} ({ecoule / 60:.0f} min déjà faites)")
    else:
        net.load_state_dict(torch.load(a.depart, map_location="cpu", weights_only=False)["net"], strict=True)
    net = net.to(dev)
    for st in opt.state.values():
        for k_, v_ in st.items():
            if torch.is_tensor(v_):
                st[k_] = v_.to(dev)
    log(f"{a.tag} : NAFNet de la chaîne, départ {a.depart}, {len(deb)} débruitées + {len(bru)} brutes + {len(apod)} web "
        f"(tirage {p_deb:.0%}/{a.part_brute:.0%}/{a.part_apod if apod else 0:.0%}), {len(ten)} tenues, "
        f"{'N2 (poids 0,5 -> 1 ; étoiles très brillantes sur nébuleuse ' + format(a.part_extra, '.0%') + ')' if a.n2 else 'N1'}, "
        f"étoiles sur {a.synthetic:.0%} des tuiles, lots {a.batch}, lr {a.lr}, {a.pas} pas (plafond {a.hours} h)")
    lots_it = iter(torch.utils.data.DataLoader(Lots(lots, a.batch, a.synthetic, 0 + it, a.n2, a.part_extra, a.part_grands), batch_size=None,
                                               num_workers=a.preparateurs, persistent_workers=True, prefetch_factor=4))
    t0, run, warm = time.time() - ecoule, None, a.warm
    while it < a.pas and (time.time() - t0) < a.hours * 3600:
        prog = min(1.0, max(it / a.pas, (time.time() - t0) / (a.hours * 3600)))
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (it + 1) / warm) * 0.5 * (1 + np.cos(np.pi * prog))
        x, t, w = (v.to(dev, non_blocking=True) for v in next(lots_it))
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
            v = T1.valider(net, V, dev) if V else {"val": float("nan"), "val_etoiles": float("nan")}
            hist.append({"pas": it, "minutes": round((time.time() - t0) / 60, 1), "train": round(run, 5),
                         **{k_: round(v_, 5) for k_, v_ in v.items()}})
            (out / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
            ck = {"model": "chain_nafnet", "net": net.state_dict(), "config": cfg, "step": it, "val": v}
            torch.save(ck, out / "snap" / f"pas_{it}.pt")
            torch.save({**ck, "opt": opt.state_dict(), "elapsed": time.time() - t0}, out / "last.pt")
            if V and (best is None or v["val_etoiles"] < best):
                best = v["val_etoiles"]; torch.save(ck, out / "best.pt")
            log(f"pas {it} : train {run:.5f}, validation {v['val']:.5f}, sous les étoiles {v['val_etoiles']:.5f}")
    torch.save({"model": "chain_nafnet", "net": net.state_dict(), "config": cfg, "step": it, "opt": opt.state_dict(),
                "elapsed": time.time() - t0}, out / "last.pt")
    log(f"fini : {it} pas en {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
