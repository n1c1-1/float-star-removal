"""Réseau de retrait d'étoiles (NAFNet) et son application par tuiles, extraits du module du même nom de FLOAT.

Rôle : donner à tous les scripts du dépôt (qui font « from astro_gpu.stages import star_removal ») le réseau, l'étirement
asinh fixe et l'application par tuiles, SANS avoir besoin de l'application FLOAT complète.
Entrées : une image (3, H, W) en unités de bruit (couleur moins son ciel, divisée par le bruit du ciel).
Sorties : la même image sans ses étoiles, mêmes unités.
Exemple : voir scripts/inferer_une_image.py. Poids du réseau : variable d'environnement MODELE_RESEAU (fichier .pt).
Le code du réseau est celui de FLOAT, recopié sans changement ; seule la gestion de la mémoire de la carte (devices) est retirée.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODEL = Path(os.environ.get("MODELE_RESEAU", "modeles/E3_step_20000.pt"))
#: étirement du réseau : il voit asinh(v / STRETCH) / SCALE, v en unités de bruit
STRETCH, SCALE = 3.0, 10.0
#: taille d'entraînement et tuiles pour une image plus petite qu'une grande tuile
NET_TILE, NET_OVERLAP = 256, 64
#: tuiles d'application : plus grandes que les plus grands halos d'étoiles
TILE, OVERLAP = 1024, 256
#: l'attention par canal moyenne sur cette fenêtre (taille d'entraînement), au lieu de la tuile entière
WINDOW = NET_TILE
BATCH = 16


class _LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / (var + 1e-6).sqrt() * self.w[:, None, None] + self.b[:, None, None]


class _LocalPool(nn.Module):
    """The channel attention's average. In training, over the whole tile, as published. Applied, over a ``window`` around
    each position when the features are larger than that (Test-time Local Converter, see WINDOW): a tile larger than the
    training size then sees what the network was trained on."""

    def __init__(self, window):
        super().__init__()
        self.window = window

    def forward(self, x):
        h, w = x.shape[2:]
        if self.training or (self.window >= h and self.window >= w):
            return x.mean((2, 3), keepdim=True)
        k1, k2 = min(self.window, h), min(self.window, w)
        s = nn.functional.pad(x.float().cumsum(2).cumsum(3), (1, 0, 1, 0))            # float32: bfloat16 cannot sum
        o = (s[:, :, k1:, k2:] + s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2]) / (k1 * k2)
        dh, dw = h - o.shape[2], w - o.shape[3]
        return nn.functional.pad(o, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2), mode="replicate").to(x.dtype)


class _NAFBlock(nn.Module):
    def __init__(self, c, window=WINDOW):
        super().__init__()
        self.n1, self.n2 = _LayerNorm2d(c), _LayerNorm2d(c)
        self.c1 = nn.Conv2d(c, 2 * c, 1)
        self.c2 = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sca = nn.Sequential(_LocalPool(window), nn.Conv2d(c, c, 1))
        self.c3 = nn.Conv2d(c, c, 1)
        self.c4 = nn.Conv2d(c, 2 * c, 1)
        self.c5 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.c2(self.c1(self.n1(inp)))
        a, b = x.chunk(2, 1)
        x = a * b
        x = self.c3(x * self.sca(x))
        y = inp + x * self.beta
        a, b = self.c4(self.n2(y)).chunk(2, 1)
        return y + self.c5(a * b) * self.gamma


class NAFNet(nn.Module):
    """NAFNet (Chen, Chu, Zhang, Sun 2022, arXiv:2204.04676), as published (github.com/megvii-research/NAFNet).
    ``recompute``: in training, the blocks' activations are recomputed on the way back instead of kept (less memory,
    about a third slower)."""

    def __init__(self, width=32, enc=(2, 2, 4, 8), mid=12, dec=(2, 2, 2, 2), recompute=False):
        super().__init__()
        self.recompute = recompute
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encs, self.downs, self.ups, self.decs = (nn.ModuleList() for _ in range(4))
        c, win = width, WINDOW                                        # win: WINDOW in this level's own pixels
        for n in enc:
            self.encs.append(nn.Sequential(*[_NAFBlock(c, win) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, 2))
            c, win = 2 * c, win // 2
        self.mid = nn.Sequential(*[_NAFBlock(c, win) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c, win = c // 2, win * 2
            self.decs.append(nn.Sequential(*[_NAFBlock(c, win) for _ in range(n)]))
        self.pad = 2 ** len(enc)

    def _run(self, m, v):
        if self.recompute and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(m, v, use_reentrant=False)
        return m(v)

    def forward(self, inp):
        H, W = inp.shape[2:]
        ph, pw = (-H) % self.pad, (-W) % self.pad
        x = nn.functional.pad(inp, (0, pw, 0, ph))
        x0 = x
        x = self.intro(x)
        skips = []
        for e, dn in zip(self.encs, self.downs):
            x = self._run(e, x)
            skips.append(x)
            x = dn(x)
        x = self._run(self.mid, x)
        for d, up, sk in zip(self.decs, self.ups, reversed(skips)):
            x = self._run(d, up(x) + sk)
        return (self.ending(x) + x0)[:, :, :H, :W]




def fwd(v):
    """The network's stretch (numpy or torch), v in noise units."""
    mod = np if isinstance(v, np.ndarray) else torch
    return mod.asinh(v / STRETCH) / SCALE


def inv(y):
    mod = np if isinstance(y, np.ndarray) else torch
    return mod.sinh(y * SCALE) * STRETCH




def network(dev) -> NAFNet:
    """Le réseau, poids lus dans MODELE_RESEAU (clé "net" du fichier), en float32."""
    if not MODEL.exists():
        raise FileNotFoundError(f"le réseau de retrait d'étoiles est introuvable : {MODEL} (définir MODELE_RESEAU)")
    net = NAFNet().to(dev)
    state = torch.load(MODEL, map_location=dev, weights_only=False)["net"]
    net.load_state_dict({k: v.float() for k, v in state.items()})
    net.eval()
    return net


def apply(net, img: np.ndarray, dev, ctx=None, tile: int = TILE, overlap: int = OVERLAP, batch: int = BATCH) -> np.ndarray:
    """Le réseau sur une image entière (3, H, W) en unités de bruit, par tuiles fondues par une fenêtre en cosinus ;
    renvoie l'image sans étoiles, mêmes unités. Une image plus petite qu'une grande tuile : tuiles de la taille d'entraînement."""
    if min(img.shape[1:]) < tile:
        tile, overlap = min(tile, NET_TILE), min(overlap, NET_OVERLAP)
    batch = max(1, batch * NET_TILE ** 2 // tile ** 2)
    with torch.no_grad():
        return _apply(net, img, dev, ctx, tile, overlap, batch)


def _apply(net, img: np.ndarray, dev, ctx: Context | None, tile: int, overlap: int, batch_size: int) -> np.ndarray:
    img = np.asarray(img, np.float32)
    C, H, W = img.shape
    if H < tile or W < tile:                                    # a tiny image: padded to one tile
        ph, pw = max(0, tile - H), max(0, tile - W)
        return _apply(net, np.pad(img, ((0, 0), (0, ph), (0, pw)), mode="reflect"), dev, ctx, tile, overlap,
                      batch_size)[:, :H, :W]
    step = tile - overlap
    out = np.zeros_like(img, np.float32)
    wsum = np.zeros((H, W), np.float32)
    win1 = np.sin(np.linspace(0, np.pi, tile + 2)[1:-1]) ** 2 + 1e-3
    win = np.outer(win1, win1).astype(np.float32)
    ys = list(range(0, H - tile + 1, step)) + ([H - tile] if (H - tile) % step else [])
    xs = list(range(0, W - tile + 1, step)) + ([W - tile] if (W - tile) % step else [])
    for i, y in enumerate(ys):
        batch = [img[:, y:y + tile, x:x + tile] for x in xs]
        for j in range(0, len(batch), batch_size):
            b = torch.from_numpy(np.ascontiguousarray(np.stack(batch[j:j + batch_size]))).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                r = net(fwd(b)).float()
            r = inv(r).cpu().numpy()
            for k, x in enumerate(xs[j:j + batch_size]):
                out[:, y:y + tile, x:x + tile] += r[k] * win
                wsum[y:y + tile, x:x + tile] += win
        if ctx is not None:
            ctx.progress(0.05 + 0.8 * (i + 1) / len(ys), f"ligne {i + 1}/{len(ys)}")
    return out / wsum
