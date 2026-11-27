# E3 manifest: a network trained on generated data only

E3 is the starting point of N5 and N7 (step 20000: `train/E3/snap/step_20000.pt`). This file copies what serves
as proof, without copying the data. The paths are given with the roots `<ESSAIS>` (trials folder),
`<SCRIPTS>` (scripts folder) and `<DONNEES>`.

## 1. Lineage of the weights

| network | start | pairs read | duration | sha256 checksum of the weight file resumed afterwards |
|---|---|---|---|---|
| E | random weights (log: "random weights") | `<ESSAIS>/starRemoval/pairs_v2` (generated) | 7 Nov, 17:20 -> | `train/E/snap/step_16000.pt`: f30192c5aacefede04085e1e5cc5106874c8076685152bf1f00a6f24e2eb9145 |
| E2 | E, step 16000 | `<ESSAIS>/starRemoval/pairs_v3` (generated) | 7 Nov, 22:28 -> | best point `train/E2/best.pt` (no checksum in this document) |
| E3 | E2, best point | `<ESSAIS>/starRemoval/pairs_v4` (generated) | 9 Nov, 01:33 -> 08:03 (36000 steps) | `train/E3/snap/step_20000.pt`: 9faf2dc4be332c9d389942078a019ce86248a0dbe8b1d4a32eb57284bdc2e129 |

In all three trainings, 60 % of the tiles additionally receive stars drawn on the fly by `train_e.synthetic_stars`
(analytic PSF, no image). The logs (`log.txt`) of E, E2 and E3 do not contain the mention "real images"
(the `--real` option of `train_e.py` was not used) and E3's `config.json` has no `real` key.
Checksums: E3's `config.json` fe0b5718a058525c6bb557e52c7b9ab53c24fedb313f17d35ba8fcb9fe64b5d7,
`history.json` 4d94926d0831e95182e928f7006b168efaf9d498b07d05ca964d1eef141030d9,
`log.txt` d5ae0c2094a90c6f393e0dd7a7992d2fe6260e9748b50c2f9b1377774ebee38e.

## 2. E3's pairs (pairs_v4): an entirely generated set

`<ESSAIS>/starRemoval/pairs_v4/manifest.json` (445091 bytes, written on 2026-11-09 at 01:31:35, 6474 s) has the sha256 checksum
**5bd7602ebb189ebdb0aed607240240fb1d75e7491239ffe922f756077b924c65**. It contains:

- `seed` 1, `n` 500 pairs per set, `size` 2048, sets `clean` and `defects` (1000 pairs: 950 for training,
  50 held out; `config.json`: "held" 25 per set);
- `reads_no_image`: **true** (written by `make_dataset.py`, whose header says that no image is read);
- `files`: 4000 sha256 checksums (4 files per pair: input and target, noisy and denoised);
- `denoiser`: checksum of the application's denoising network, 2ea85a0755bfccd046e32d8f9d7110896d2a9871ab499bbb9e7cd62f7a216bdc;
- `versions`: python 3.12.13, numpy 2.5.2, scipy 1.18.1, opencv 5.0.0, torch 2.11.0+cu128;
- `code`: checksums of the code that generates the pairs:

| file | sha256 |
|---|---|
| make_dataset.py | cd57068d1a3eee14007124984562e9421b1053ded0e36f6334cf0b9edc006654 |
| backgrounds.py | e627d48df58f6f8e64a7ecc358eee80c5b6bb1fc4d05e0c8bbccbff944fc6e1f |
| defects.py | ed5729101c4090eef2091c7f476a220392165893efd782db0e4ff225750cbfff |
| psf.py | 84a3d3459b1851578a71349efe407478ccfb5477d56820b58c7a3830037224b1 |
| instruments.py | e1de75dd1ab5d0ee6f1b2f25c5022a616a674aa2546b8d28f6ce08f7b398a9cd |
| stars.py | acfadf33ed7aa129a31e7363cd49ded14c624e37102f6d1d9a7a9f2f5eaa13e1 |
| catalogue.json | bccd5e1ab07de7a4172c0acce0d91e642d5a1ec8bb66e27121c4c8e31d9969cd |

Verification done on 26 Nov 2026: the checksums of the current files in `<SCRIPTS>` (these seven files) are
**identical** to those of the manifest. The code that generated E3's pairs is therefore, byte for byte, the code present here.

What the generator (`make_dataset.py`) reads: `catalogue.json` (published characteristics of telescopes, lenses and
cameras) and the application's denoising network. The backgrounds (`backgrounds.py`) and the defects (`defects.py`) are computed by
code from the seed; the stars by `stars.py` and `psf.py` (Fourier optics). Real images were used
only to measure drawing ranges (constants in the code, with the measurement they come from).

## 3. The other generated sets of the lineage

| set | manifest.json sha256 | written on | backgrounds.py / stars.py / make_dataset.py in the manifest |
|---|---|---|---|
| pairs_v2 (E) | c48a66e416769318dd6f1afca54411fa3222c54d262fc068f8475524646ab939 | 2026-11-07 10:25 | 539ff547... / ec671b7b... / c76376b4... |
| pairs_v3 (E2) | 1ced1a48e728b821042f37588e86130e58f92984833e3eb0ace46ea358688cea | 2026-11-07 22:21 | e627d48d... / ec671b7b... / 3374b83e... |
| pairs_v4 (E3) | 5bd7602ebb189ebdb0aed607240240fb1d75e7491239ffe922f756077b924c65 | 2026-11-09 01:31 | e627d48d... / acfadf33... / cd57068d... |

The three manifests carry `reads_no_image: true`. For pairs_v2 the code has changed since (checksums differ from
those of the present code for backgrounds.py, stars.py and make_dataset.py); for pairs_v3, stars.py and make_dataset.py have
changed since: the exact code of these two sets is not kept; only the manifest keeps its checksums. The PSF bank
`psf_bank.npz` (sha256 6075a9da6a3aee6b4ddd9ad4b2a0397fcfd58ecca66bb212ff7680288f412856) is computed by
`synth.py build` from the optical model and drawn instruments (seed 2718): no image.

## 4. How to redo it

`make_dataset.py --verify 1 --out <folder>` regenerates the first pair of each set from the seed and compares the
checksums to the manifest: noisy files byte for byte; denoised files byte for byte on CPU, or within a tolerance if the
denoising ran on a graphics card (the case of pairs_v4: `device` cuda:0). Two check regenerations of pairs_v4 exist in the
logs of the time (`pairs_v4_check.log`, `pairs_v4_check2.log`: 240 files each, ~8 min); they display only the regenerated
pairs, not the verdict of the comparison: **that old verdict is not kept**.

Verification redone on 26 Nov 2026 (`make_dataset.py --verify 1`, one pair per set: `clean/pair_1_0000` and `defects/pair_1_0000`,
regenerated on the CPU for the generation and on the graphics card for the denoising, as the manifest says):

| file | result |
|---|---|
| `input_noisy.npy`, `target_noisy.npy` (both sets) | **bit-identical** to the manifest's checksum |
| `input_denoised.npy`, `target_denoised.npy` (both sets) | **different**: largest difference 216 (input) and 88 (target), in noise units, in a 2048 x 2048 float16 image, beyond the script's tolerance (0.05) |

Reading: everything drawn from the seed (backgrounds, defects, stars, noise: the "noisy" files) is reproduced exactly by the present
code. The "denoised" files are these same images passed through the application's **denoising** network; the stored files do have
the manifest's checksum, and the current denoiser's weight file has the same checksum as the manifest's, but the code of the engine's
denoising stage was modified after 9 Nov (commits of 11, 12, 13 and 15 Nov, including "deterministic seed in denoising").
**Probable cause, not established**: this code change, or the non-determinism of the graphics card. This difference concerns denoising, not star
removal: it does not change the origin of the pairs (generated).

## 5. What remains uncertain for E3 (by name)

- `train_e.py`, the training script, was modified on 9 Nov at 15:24, after the end of E3's training
  (08:03): the exact version launched is not fixed by a checksum. The visible modification is the addition of the option
  `--real` (never used for E, E2, E3). The scripts' git repository exists only since 21 Nov: no commit dates
  from E3's training.
- The point `train/E2/best.pt` has no checksum in this document.
- `day_e4.py` (E4: E3 + corrected real pairs from `real_targets.py`) exists, but E4 was never trained
  (no `train/E4` folder); N5 and N7 start from E3 step 20000, not from E4.
