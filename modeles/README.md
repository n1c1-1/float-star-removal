# The models

Four NAFNet networks (width 32, ≈ 29 million parameters in float32; architecture: `scripts/astro_gpu/stages/star_removal.py`). **Each file is over 90 MB: none is in git.** They are release attachments (Releases) of the repository. The checksums are in `SHA256SUMS`:

```
sha256sum -c SHA256SUMS
```

| file | size (bytes) | sha256 | what it is | when to use it |
|---|---|---|---|---|
| `E3_step_20000.pt` | 116 862 057 | `9faf2dc4…e2129` | E3, step 20,000: trained on **generated data only** (8-9 Nov); starting point of N5 and N7 | trial, or starting point of a new training; worse on real data |
| `N5_pas_23000.pt` | 116 868 111 | `d4b0309c…f9d6` | N5, step 23,000 (best validation step): E3 fine-tuned on real v33g pairs (25 Nov) | **fields without galaxy core or compact planetary**: also removes saturated stars of low signal (with two-scale inference); **erases galaxy cores** |
| `N7_pas_22000.pt` | 116 868 111 | `746ce23e…5c8` | N7, step 22,000: N5 + synthetic galaxy cores (25 Nov) | when **galaxy cores must be kept**; keeps some saturated stars of low signal |
| `N7N5_075.pt` | 116 835 023 | `9c88c524…199b` | weight blend 0.75 × N7 + 0.25 × N5 (26 Nov) | the batch's default choice; behaves like N7 on saturated stars of low signal |

(full checksums in `SHA256SUMS`.) The four files are `torch` dictionaries with the keys `net` (float32 weights; this is what `star_removal.network()` and `scripts/inferer_une_image.py` read) and `step`; E3 and N5/N7 also have `model`, `config` and `val`, the blend has `melange` (its two sources and its coefficient).

## How to choose (a trade-off, not a best model)

| | galaxy cores kept | saturated stars of low signal removed | faint stars G 14-17 remaining |
|---|---|---|---|
| N5 (two scales) | no (0.19) | yes | 0.265 |
| N7 | yes (0.81) | partly no | 0.283 |
| N7/N5 0.75 | yes (0.80) | partly no | 0.295 |

See `README.md` (results table) and `docs/IMPROVING.md` (a lead to lift this trade-off).

## Origin and provenance

All come from trainings published in this repository (`scripts/`, `prompts/`), on data whose provenance is established file by file (`docs/PROVENANCE.md`, `provenance/`). **None was trained on the output of another star-removal program.**

## Licence

AGPL-3.0, like the rest of the repository (`LICENSE`, `NOTICE`). The weights are part of the Program covered by that licence.
