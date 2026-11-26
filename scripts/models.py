# ======================================================================
# models.py
# Rôle     : Définition des réseaux (NAFNet...), étirement asinh fixe, application par tuiles.
# Étape    : 1. Entraînement sur données synthétiques seules (E2 / E3, 7-9 nov) (voir scripts/LISEZMOI.md)
# Entrées  : générateur : graines + catalogue.json (fiches d'instruments publiques) ; aucune image
# Sorties  : paires synthétiques (input/target/info.json + manifest.json), réseau E3
# Exemple  : python models.py [arguments]  (voir la docstring ; chemins par variables RACINE_SCRIPTS, DONNEES, ESSAIS)
# ======================================================================
"""The star removal networks of the bench: their definitions, the stretch they work in, and their tiled application.

- ``nafnet``: NAFNet (Chen, Chu, Zhang, Sun 2022, ECCV, arXiv:2204.04676), width 32, as published; it predicts the
  image without its stars (the input plus a correction). Its channel attention averages over the whole tile: applied
  to tiles larger than its training patches, the average is taken over a window of the training size instead (TLC:
  Chu et al. 2022, ECCV, arXiv:2112.04491), so that it behaves as in training;
- ``nafnet_mask``: the same network predicting the star layer and a soft mask; the output is the input minus
  mask x stars: off the stars the input comes back unchanged, its noise included;
- ``lama``: LaMa's network (Suvorov et al. 2022, WACV, arXiv:2109.07161): a ResNet of fast Fourier convolutions
  (Chi et al. 2020) whose receptive field covers the whole tile, made for filling large holes, robust to resolutions
  larger than the training one; here residual (the input plus a correction).

All see the data through a fixed stretch, asinh(v / 3) / 10 (v in the noise units of the pairs), undone exactly.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STRETCH, SCALE = 3.0, 10.0


def fwd(v):
    mod = np if isinstance(v, np.ndarray) else torch
    return mod.asinh(v / STRETCH) / SCALE


def inv(y):
    mod = np if isinstance(y, np.ndarray) else torch
    return mod.sinh(y * SCALE) * STRETCH


# ---------------------------------------------------------------------------------------------- NAFNet
class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / (var + 1e-6).sqrt() * self.w[:, None, None] + self.b[:, None, None]


def local_mean(x: torch.Tensor, k: int) -> torch.Tensor:
    """The mean over a k x k window around each pixel (an integral image: any k at the same cost), edges included."""
    B, C, H, W = x.shape
    k = min(k, H, W)
    r = k // 2
    s = F.pad(x, (r + 1, r, r + 1, r), mode="replicate").cumsum(2).cumsum(3)
    tot = s[:, :, k:, k:] - s[:, :, :-k, k:] - s[:, :, k:, :-k] + s[:, :, :-k, :-k]
    return tot[:, :, :H, :W] / (k * k)


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = LayerNorm2d(c), LayerNorm2d(c)
        self.c1 = nn.Conv2d(c, 2 * c, 1)
        self.c2 = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sca = nn.Conv2d(c, c, 1)
        self.c3 = nn.Conv2d(c, c, 1)
        self.c4 = nn.Conv2d(c, 2 * c, 1)
        self.c5 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.tlc = 0                                   # set at inference: the training window at this scale (TLC)

    def forward(self, inp):
        x = self.c2(self.c1(self.n1(inp)))
        a, b = x.chunk(2, 1)
        x = a * b
        if self.tlc and (x.shape[-1] > self.tlc or x.shape[-2] > self.tlc):
            pooled = local_mean(x, self.tlc)
        else:
            pooled = x.mean((2, 3), keepdim=True)
        x = self.c3(x * self.sca(pooled))
        y = inp + x * self.beta
        a, b = self.c4(self.n2(y)).chunk(2, 1)
        return y + self.c5(a * b) * self.gamma


class NAFNet(nn.Module):
    def __init__(self, width=32, enc=(2, 2, 4, 8), mid=12, dec=(2, 2, 2, 2), out_ch=3):
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, out_ch, 3, padding=1)
        self.encs, self.downs, self.ups, self.decs = (nn.ModuleList() for _ in range(4))
        c = width
        for n in enc:
            self.encs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, 2))
            c *= 2
        self.mid = nn.Sequential(*[NAFBlock(c) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.pad = 2 ** len(enc)

    def set_tlc(self, train_patch: int):
        """TLC: each block's attention averages over the training patch's size at its own scale."""
        n = len(self.encs)
        levels = [(e, 2 ** i) for i, e in enumerate(self.encs)] + [(self.mid, 2 ** n)] + \
            [(d, 2 ** (n - 1 - j)) for j, d in enumerate(self.decs)]           # the decoder: deepest first
        for blocks, scale in levels:
            for b in blocks:
                b.tlc = max(train_patch // scale, 1)

    def features(self, inp):
        H, W = inp.shape[2:]
        ph, pw = (-H) % self.pad, (-W) % self.pad
        x = F.pad(inp, (0, pw, 0, ph))
        x = self.intro(x)
        skips = []
        for e, dn in zip(self.encs, self.downs):
            x = e(x)
            skips.append(x)
            x = dn(x)
        x = self.mid(x)
        for d, up, sk in zip(self.decs, self.ups, reversed(skips)):
            x = d(up(x) + sk)
        return self.ending(x)[:, :, :H, :W]

    def forward(self, inp):
        return self.features(inp) + inp


class NAFNetMask(NAFNet):
    """NAFNet predicting the star layer S (>= 0, in the stretch) and a mask M: output = input - M x S."""

    def __init__(self, **kw):
        super().__init__(out_ch=4, **kw)

    def forward(self, inp, with_mask: bool = False):
        f = self.features(inp)
        stars, mask = F.softplus(f[:, :3]), f[:, 3:4]
        out = inp - torch.sigmoid(mask) * stars
        return (out, mask) if with_mask else out


# ---------------------------------------------------------------------------------------------- LaMa
class Spectral(nn.Module):
    """LaMa's spectral transform: a 1 x 1 convolution on the real and imaginary parts of the tile's FFT."""

    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(c, c // 2, 1, bias=False), nn.BatchNorm2d(c // 2), nn.ReLU())
        self.fu = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.c2 = nn.Conv2d(c // 2, c, 1, bias=False)

    def forward(self, x):
        x = self.c1(x)
        h, w = x.shape[-2:]
        f = torch.fft.rfft2(x.float(), norm="ortho")
        f = self.fu(torch.cat([f.real, f.imag], 1).to(x.dtype))
        re, im = f.float().chunk(2, 1)
        y = torch.fft.irfft2(torch.complex(re, im), s=(h, w), norm="ortho").to(x.dtype)
        return self.c2(x + y)


class FFC(nn.Module):
    def __init__(self, c, ratio=0.75):
        super().__init__()
        g = int(c * ratio)
        lo = c - g
        self.lo = lo
        self.l2l = nn.Conv2d(lo, lo, 3, padding=1, bias=False)
        self.l2g = nn.Conv2d(lo, g, 3, padding=1, bias=False)
        self.g2l = nn.Conv2d(g, lo, 3, padding=1, bias=False)
        self.g2g = Spectral(g)
        self.bl, self.bg = nn.BatchNorm2d(lo), nn.BatchNorm2d(g)

    def forward(self, x):
        xl, xg = x[:, :self.lo], x[:, self.lo:]
        yl = F.relu(self.bl(self.l2l(xl) + self.g2l(xg)))
        yg = F.relu(self.bg(self.l2g(xl) + self.g2g(xg)))
        return torch.cat([yl, yg], 1)


class FFCRes(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a, self.b = FFC(c), FFC(c)

    def forward(self, x):
        return x + self.b(self.a(x))


class LaMa(nn.Module):
    def __init__(self, ngf=64, down=3, blocks=9):
        super().__init__()
        L = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.BatchNorm2d(ngf), nn.ReLU()]
        c = ngf
        for _ in range(down):
            L += [nn.Conv2d(c, 2 * c, 3, 2, 1), nn.BatchNorm2d(2 * c), nn.ReLU()]
            c *= 2
        L += [FFCRes(c) for _ in range(blocks)]
        for _ in range(down):
            L += [nn.ConvTranspose2d(c, c // 2, 3, 2, 1, output_padding=1), nn.BatchNorm2d(c // 2), nn.ReLU()]
            c //= 2
        L += [nn.ReflectionPad2d(3), nn.Conv2d(c, 3, 7)]
        self.net = nn.Sequential(*L)
        self.pad = 2 ** down

    def forward(self, inp):
        H, W = inp.shape[2:]
        ph, pw = (-H) % self.pad, (-W) % self.pad
        x = F.pad(inp, (0, pw, 0, ph), mode="reflect")
        return self.net(x)[:, :, :H, :W] + inp


MODELS = {"nafnet": NAFNet, "nafnet_mask": NAFNetMask, "lama": LaMa}
#: the tile each model is applied on, and the overlap (NAFNet with TLC can take tiles larger than its training size;
#: LaMa sees the whole tile at once: large tiles let it see the halos)
INFER = {"nafnet": (512, 64), "nafnet_mask": (512, 64), "lama": (1024, 128), "chain_nafnet": (256, 64)}


def build(name: str) -> nn.Module:
    return MODELS[name]()


def chain_nafnet():
    """The chain's own NAFNet (astro_gpu.stages.star_removal: the definition, the stretch and the 256 px tiles of the
    network it ships), for train_e.py's runs."""
    import sys
    sys.path.insert(0, "<FLOAT>")
    from astro_gpu.stages.star_removal import NAFNet as ChainNAFNet
    return ChainNAFNet()


def load(path, device) -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("model") == "chain_nafnet":
        net = chain_nafnet().to(device)
        net.load_state_dict(ck["net"])
        net.eval()
        return net, ck["config"]
    net = build(ck["model"]).to(device)
    net.load_state_dict(ck["net"])
    net.eval()
    if isinstance(net, NAFNet):
        net.set_tlc(ck["config"]["patch"])
    return net, ck["config"]


def apply(net, img: np.ndarray, device, tile: int, overlap: int) -> np.ndarray:
    """A network over a whole (3, H, W) image in noise units, in tiles blended with a cosine window; the image without
    its stars, in the same units."""
    dev = torch.device(device) if isinstance(device, str) else device
    img = np.asarray(img, np.float32)
    C, H, W = img.shape
    tile = min(tile, H, W) // 8 * 8
    step = max(tile - overlap, 8)
    out = np.zeros_like(img)
    wsum = np.zeros((H, W), np.float32)
    w1 = np.sin(np.linspace(0, np.pi, tile + 2)[1:-1]) ** 2 + 1e-3
    win = np.outer(w1, w1).astype(np.float32)
    ys = list(range(0, H - tile + 1, step)) + ([H - tile] if (H - tile) % step else [])
    xs = list(range(0, W - tile + 1, step)) + ([W - tile] if (W - tile) % step else [])
    for y in ys:
        for x in xs:
            b = torch.from_numpy(np.ascontiguousarray(img[None, :, y:y + tile, x:x + tile])).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                r = net(fwd(b))
            r = inv(r.float()).cpu().numpy()[0]
            out[:, y:y + tile, x:x + tile] += r * win
            wsum[y:y + tile, x:x + tile] += win
    return out / wsum
apply = torch.no_grad()(apply)
