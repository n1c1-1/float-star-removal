# ======================================================================
# train_n9.py
# Role     : N9 = copy of train_n8l.py (N7 + memory guard and core check inside the process).
# Stage    : 10. Discarded trial N9 (compact cores, 26 Nov) (see scripts/LISEZMOI.md)
# Inputs   : v33g pairs
# Outputs  : snapshots, tableau_n9.md
# Example  : python train_n9.py [arguments]  (see the docstring; paths via variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""N9 = copy of train_n8l.py (N7 + memory guard and core check inside the process). Recipe change: 40 % compact synthetic cores
(paires_synthetiques_n7.PART_COMPACTS). Loaders 2 / prefetch 4 as N7; core check at steps 5000 and 10000.
COPY of train_n8.py (N8b): resume from step 5000 (last.pt: network, optimiser, step; the learning step follows from the step, same formula) with reduced memory
(--preparateurs 1, prefetch 2, last.pt released after reading) and memory guard every 100 steps (clean stop under 3.5 GB free, code 4).
COPY of train_n7.py for N8 (only changes: --coeurs 0,10 given by the chain, --pas 10000, core measurement at step 5000 with stop rule).
COPY of train_n56.py for N7: exact N5/N6 recipe; only change, --coeurs (0.25): on this share of the synthetic tiles,
1 or 2 extended cores (paires_synthetiques_n7.paire_synthetique_coeurs) present identically in the input and the target, weight 1 on them.

Night trials of 24 Nov (N1, N2): V33C (train_v33c.py, itself A) with the v33d pairs and the APOD web pairs.

Imports train_jeu1.py WITHOUT modifying it (chain_nafnet network, weighted L1 loss, validation). Start E3 step 20000, batches 8, lr 5e-4,
1000-step warm-up then cosine over the steps, 60 % synthetic-star tiles. No artificial noise added.

N1 = V33C with the new data, nothing else:
  - real pairs <DONNEES>\\Paires étoiles v33 – 23 nov\\*_v33d.npz (denoised and raw), same 6 held-out tiles as V33C;
  - web pairs <DONNEES>\\Paires étoiles APOD – 24 nov\\*_v33d.npz (denoised only);
  - draw: 50 % real denoised, 25 % raw, 25 % APOD (if no APOD pair: the refusal is explicit, see --sans-apod).
N2 = N1 plus two changes (--n2):
  (a) the weights between 0 and 1 (0.5: repair judged good) become 1 in all training pairs; the validation
      during the run keeps the original weights (comparable to N1);
  (b) on --part-extra (0.5) of the synthetic tiles, one or two imposed very bright stars (peak 300 to 3e5 noises, with
      the generator's large halo and spikes), placed on a real NEBULA background: a place where the target has weight 1 over a
      40 px disk (90 %) and where its smoothed level exceeds 1 noise (otherwise no imposed star).
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

import gc
import cv2
import numpy as np
import psutil
import torch

ICI = Path(__file__).resolve().parent
RACINE = (Path(__import__("os").environ.get("RACINE_SCRIPTS") or Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "Entrainement jeu 1 – 22 nov"))
sys.path.insert(0, str(RACINE / "Etoiles synthetiques – 22 nov"))
sys.path.insert(0, str(ICI))
from models import chain_nafnet, fwd  # noqa: E402
import train_jeu1 as T1  # noqa: E402

PAIRES = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles v33 – 23 nov")
APOD = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles APOD – 24 nov")
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles N9 – 26 nov")
SUF = "v33g"        # N5 ; N6 : --suffixe v33h
MOT = re.compile(r"(.+_g\d+_\d+)_(debruitee|brute)_v33[dgh]$")
TENUES = ("m13_g1408_1408,m57_g1664_2688,ngc6888_g896_1408,ngc7000_tiers1_g2432_1920,"
          "rosette_40da_g1920_1920,wizard_a7s_g2944_1664")


def lister(tenues, apod_dir=APOD):
    deb, bru, ten = [], [], []
    for f in sorted(PAIRES.glob(f"*_{SUF}.npz")):
        m = MOT.match(f.stem)
        assert m, f.name
        if m.group(1) in tenues:
            if m.group(2) == "debruitee":
                ten.append(f)
        elif m.group(2) == "debruitee":
            deb.append(f)
        else:
            bru.append(f)
    apod = [f for f in sorted(Path(apod_dir).glob(f"*_{SUF}.npz")) if "_brute" not in f.stem]
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


class Lots(torch.utils.data.IterableDataset):
    def __init__(self, lots, batch, synthetic, seed, n2, part_extra, p_coeurs=0.0):
        self.lots, self.batch, self.synthetic, self.seed = lots, batch, synthetic, seed
        self.n2, self.part_extra, self.p_coeurs = n2, part_extra, p_coeurs

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        if info:
            torch.set_num_threads(1)
        from paires_synthetiques_n7 import paire_synthetique_coeurs
        D = Melange(self.lots, self.n2, self.seed + 1000 * wid)
        rng = np.random.default_rng(self.seed + 7 + 1000 * wid)
        k = 0
        while True:
            x, t, w, _, _, _ = D.get(self.batch)
            for i in range(self.batch):
                if rng.random() < self.synthetic:
                    fond = t[i] if rng.random() < 0.5 else x[i]
                    ex = extras_nebuleuse(t[i], w[i], rng) if (self.n2 and rng.random() < self.part_extra) else None
                    e, _, co, inf = paire_synthetique_coeurs(np.ascontiguousarray(fond, np.float32), int(rng.integers(2**31)) + k,
                                                             extras=ex, p_coeurs=self.p_coeurs)
                    x[i] = e.astype(np.float32)
                    if inf:                                  # cœurs : gardés dans la cible, poids 1 là où ils comptent
                        t[i] = t[i] + co
                        w[i] = np.maximum(w[i], (co.max(0, keepdims=True) > 0.05).astype(np.float32))
                k += 1
            yield torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(w)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--suffixe", default="v33g", help="paires : v33g (N5) ou v33h (N6)")
    ap.add_argument("--n2", action="store_true", help="(a) poids 0,5 -> 1 et (b) étoiles très brillantes sur fonds de nébuleuse")
    ap.add_argument("--part-extra", type=float, default=0.5)
    ap.add_argument("--coeurs", type=float, default=0.25, help="N7 : part des tuiles synthétiques qui reçoivent 1-2 cœurs étendus gardés")
    ap.add_argument("--controle-pas", default="5000,10000", help="N9 : pas où l'on mesure les cœurs (règle d'arrêt), séparés par des virgules")
    ap.add_argument("--controle-min", type=float, default=0.6, help="N8 : arrêt si le rapport moyen des cœurs est inférieur")
    ap.add_argument("--snap-pas", type=int, default=1000)
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
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--garde-go", type=float, default=3.5, help="arrêt mémoire si la mémoire libre passe sous ce seuil (contrôle tous les 100 pas)")
    ap.add_argument("--limite", type=int, default=0, help="essai à blanc : n paires de chaque sorte")
    ap.add_argument("--depart", default=r"<ESSAIS>\starRemoval\train\E3\snap\step_20000.pt")
    ap.add_argument("--sortie", default=str(SORTIE))
    ap.add_argument("--reprendre", action="store_true")
    a = ap.parse_args(argv)
    global SUF
    SUF = a.suffixe
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

    deb, bru, apod, ten = lister({s for s in a.tenues.split(",") if s}, a.apod_dir)
    if not apod and not a.sans_apod:
        sys.exit("aucune paire web *_v33d.npz dans " + a.apod_dir + " (--sans-apod pour s'en passer)")
    if a.limite:
        deb, bru, apod = deb[:a.limite], bru[:a.limite], apod[:a.limite]
    p_deb = 1.0 - a.part_brute - (a.part_apod if apod else 0.0)
    lots = [("debruitees", p_deb, deb), ("brutes", a.part_brute, bru), ("apod", a.part_apod, apod)]
    cfg = {**vars(a), "n_debruitees": len(deb), "n_brutes": len(bru), "n_apod": len(apod), "n_tenues": len(ten),
           "debruitees": [f.stem for f in deb], "brutes": [f.stem for f in bru], "apod": [f.stem for f in apod],
           "tenues_fichiers": [f.stem for f in ten],
           "recette": "train_v33c.py (A_E3) ; exemples v33d + APOD ; perte L1 pondérée ; aucun bruit ajouté"}
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
        del ck; gc.collect()
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
    lots_it = iter(torch.utils.data.DataLoader(Lots(lots, a.batch, a.synthetic, 0 + it, a.n2, a.part_extra, a.coeurs), batch_size=None,
                                               num_workers=a.preparateurs, persistent_workers=a.preparateurs > 0,
                                               **({"prefetch_factor": a.prefetch} if a.preparateurs > 0 else {})))
    t0, run, warm = time.time() - ecoule, None, 1000
    arret_mem = False
    controles = {int(x) for x in str(a.controle_pas).split(',') if x}
    while it < a.pas and (time.time() - t0) < a.hours * 3600:
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (it + 1) / warm) * 0.5 * (1 + np.cos(np.pi * it / a.pas))
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
        if it % a.snap_pas == 0:
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
            if it in controles:                           # N8L : colonne cœurs aux pas de contrôle, même code que la chaîne N7
                import coeurs_mesure as CM
                CM.OUT = Path(a.sortie)
                CM.mesurer(f"{a.tag}_{it}", str(out / "snap" / f"pas_{it}.pt"), a.device)
                moy = json.loads((Path(a.sortie) / "coeurs" / f"coeurs_{a.tag}_{it}.json").read_text(encoding="utf-8"))["moyenne"]
                m32 = [c for c in json.loads((Path(a.sortie) / "coeurs" / f"coeurs_{a.tag}_{it}.json").read_text(encoding="utf-8"))["coeurs"]
                       if c["image"] == "m31_canon60d_objectif_filee" and (c["x"], c["y"]) == (3094, 1704)][0]["rapport"]
                log(f"pas {it} : cœurs (sortie/entrée) {moy:.3f} ; M32 Canon 60D seul {m32:.3f} ; seuil d'arrêt {a.controle_min}")
                if moy < a.controle_min:
                    log("arrêt : cœurs trop bas")
                    (out / "arret_coeurs.txt").write_text(f"arrêt : cœurs trop bas ({moy:.3f} < {a.controle_min}) au pas {it}", encoding="utf-8")
                    break
        if it % 100 == 0 and psutil.virtual_memory().available / 2**30 < a.garde_go:
            log(f"arrêt mémoire : {psutil.virtual_memory().available / 2**30:.2f} Go libres < {a.garde_go} Go au pas {it}")
            arret_mem = True
            break
    torch.save({"model": "chain_nafnet", "net": net.state_dict(), "config": cfg, "step": it, "opt": opt.state_dict(),
                "elapsed": time.time() - t0}, out / "last.pt")
    if arret_mem:
        log(f"last.pt sauvegardé au pas {it}, arrêt propre")
        sys.exit(4)
    if (out / "arret_coeurs.txt").exists() and it < a.pas:
        log(f"arrêté au pas {it} (cœurs trop bas)")
        return
    log(f"fini : {it} pas en {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
