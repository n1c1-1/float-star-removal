# ======================================================================
# train_v33c.py
# Rôle     : Essai de 2 h sur les paires v33c (24 nov), mêmes conditions que l'essai A du 22 nov (A_E3).
# Étape    : 6. Entraînement d'essai 2 h (24 nov) (voir scripts/LISEZMOI.md)
# Entrées  : paires v33c
# Sorties  : instantanés, tableau de mesure
# Exemple  : python train_v33c.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""Essai de 2 h sur les paires v33c (24 nov), mêmes conditions que l'essai A du 22 nov (A_E3).

Importe train_jeu1.py SANS le modifier : même réseau (chain_nafnet), même perte L1 pondérée par les poids des paires, mêmes
60 % de tuiles d'étoiles synthétiques v2 (sur fond propre ou entrée), mêmes lots de 8, lr 5e-4, montée 1000 pas puis cosinus,
départ E3 pas 20000. Différences voulues, et seulement celles-ci :
- les exemples : <DONNEES>\\Paires étoiles v33 – 23 nov\\*_v33c.npz, versions débruitée ET brute de la même tuile ;
- le tirage : part de tuiles brutes fixée (--part-brute, 0,3) ; les brutes n'existent que pour 12 images ;
- le calendrier du cosinus suit les pas (--pas 13400, vitesse de A : 13 439 pas en 120 min), pas l'horloge ;
- aucun bruit artificiel ajouté (interdit) : les entrées sont celles des paires, et les étoiles synthétiques gardent le seul
  bruit de photon de paires_synthetiques.py comme en A.
Les tenues (validation en cours de route) sont des tuiles entières : débruitée ET brute écartées de l'entraînement.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "Entrainement jeu 1 – 22 nov"))
sys.path.insert(0, str(RACINE / "Etoiles synthetiques – 22 nov"))
from models import chain_nafnet, fwd  # noqa: E402
import train_jeu1 as T1  # noqa: E402

PAIRES = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Paires étoiles v33 – 23 nov")
SORTIE = (Path(__import__("os").environ.get("DONNEES", "donnees")) / "Entrainement étoiles v33c – 24 nov")
PATCH = T1.PATCH
MOT = re.compile(r"(.+_g\d+_\d+)_(debruitee|brute)_v33c$")


def lister(tenues):
    """Retourne (débruitées, brutes, tenues_débruitées) ; une tuile tenue sort dans ses deux versions."""
    deb, bru, ten = [], [], []
    # 6 paires que le juge v33c n'a pas modifiées : leur ancien fichier (sans _v33c) vaut v33c ; les lâchées restent écartées
    import csv
    lach = {(r[0], r[1]) for r in csv.reader(open(PAIRES / "revue_orchestrateur.csv", encoding="utf-8"), delimiter=";")
            if len(r) > 2 and r[2] == "lâchée"}
    fich = sorted(PAIRES.glob("*_v33c.npz"))
    for f in sorted(PAIRES.glob("*.npz")):
        if f.stem.endswith("_v33c") or (f.with_name(f.stem + "_v33c.npz")).exists():
            continue
        m = re.match(r"(.+?)_(g\d+_\d+)_(?:debruitee|brute)$", f.stem)
        if m and (m.group(1), "*") not in lach and (m.group(1), m.group(2)) not in lach:
            fich.append(f)
    for f in fich:
        m = MOT.match(f.stem) or re.match(r"(.+_g\d+_\d+)_(debruitee|brute)$", f.stem)
        assert m, f.name
        if m.group(1) in tenues:
            if m.group(2) == "debruitee":
                ten.append(f)
        elif m.group(2) == "debruitee":
            deb.append(f)
        else:
            bru.append(f)
    return deb, bru, ten


class Melange:
    """Même interface que T1.Paires.get ; chaque tuile vient du lot brut avec la probabilité part_brute."""

    def __init__(self, deb, bru, part_brute, seed):
        self.D = T1.Paires(deb, seed)
        self.B = T1.Paires(bru, seed + 1) if bru else None
        self.part, self.rng = part_brute, np.random.default_rng(seed + 2)

    def get(self, n):
        parts = [(self.B if (self.B and self.rng.random() < self.part) else self.D).get(1) for _ in range(n)]
        return tuple(np.concatenate([p[j] for p in parts]) for j in range(6))


class Lots(T1.LotsSynth):
    def __init__(self, deb, bru, part_brute, batch, synthetic, seed):
        super().__init__(deb, batch, synthetic, seed)
        self.bru, self.part = bru, part_brute

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        torch.set_num_threads(1)
        from paires_synthetiques import paire_synthetique
        D = Melange(self.fichiers, self.bru, self.part, self.seed + 1000 * wid)
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="V33C")
    ap.add_argument("--pas", type=int, default=13400, help="nombre de pas (calendrier du cosinus)")
    ap.add_argument("--hours", type=float, default=2.4, help="plafond de sécurité en heures")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--synthetic", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--part-brute", type=float, default=0.3)
    ap.add_argument("--tenues", default="m13_g1408_1408,m57_g1664_2688,ngc6888_g896_1408,ngc7000_tiers1_g2432_1920,"
                    "rosette_40da_g1920_1920,wizard_a7s_g2944_1664")
    ap.add_argument("--preparateurs", type=int, default=2)
    ap.add_argument("--depart", default=r"<ESSAIS>\starRemoval\train\E3\snap\step_20000.pt")
    ap.add_argument("--limite", type=int, default=0, help="essai à blanc : n paires de chaque sorte seulement")
    ap.add_argument("--sortie", default=str(SORTIE))
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

    deb, bru, ten = lister({s for s in a.tenues.split(",") if s})
    if a.limite:  # essai à blanc seulement
        deb, bru = deb[:a.limite], bru[:a.limite]
    cfg = {**vars(a), "n_debruitees": len(deb), "n_brutes": len(bru), "n_tenues": len(ten),
           "debruitees": [f.stem for f in deb], "brutes": [f.stem for f in bru], "tenues_fichiers": [f.stem for f in ten],
           "recette": "train_jeu1.py (A_E3) ; exemples = v33c ; perte L1 pondérée ; aucun bruit ajouté"}
    (out / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    V = T1.Paires(ten, 1) if ten else None
    torch.manual_seed(0)
    net = chain_nafnet()
    net.load_state_dict(torch.load(a.depart, map_location="cpu", weights_only=False)["net"], strict=True)
    net = net.to(dev)
    log(f"{a.tag} : NAFNet de la chaîne, départ {a.depart}, {len(deb)} débruitées + {len(bru)} brutes (part brute {a.part_brute:.0%}), "
        f"{len(ten)} tenues, étoiles sur {a.synthetic:.0%} des tuiles, lots {a.batch}, lr {a.lr}, {a.pas} pas (plafond {a.hours} h)")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.9))
    lots = iter(torch.utils.data.DataLoader(Lots(deb, bru, a.part_brute, a.batch, a.synthetic, 0), batch_size=None,
                                            num_workers=a.preparateurs, persistent_workers=True, prefetch_factor=4))
    hist, best = [], None
    t0, it, run, warm = time.time(), 0, None, 1000
    while it < a.pas and (time.time() - t0) < a.hours * 3600:
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (it + 1) / warm) * 0.5 * (1 + np.cos(np.pi * it / a.pas))
        x, t, w = (v.to(dev, non_blocking=True) for v in next(lots))
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
            torch.save(ck, out / "last.pt"); torch.save(ck, out / "snap" / f"pas_{it}.pt")
            if V and (best is None or v["val_etoiles"] < best):
                best = v["val_etoiles"]; torch.save(ck, out / "best.pt")
            log(f"pas {it} : train {run:.5f}, validation {v['val']:.5f}, sous les étoiles {v['val_etoiles']:.5f}")
    torch.save({"model": "chain_nafnet", "net": net.state_dict(), "config": cfg, "step": it}, out / "last.pt")
    log(f"fini : {it} pas en {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
