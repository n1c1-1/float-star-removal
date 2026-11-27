Folder names keep the original French and date for traceability.

# CPU measurement of the E3 network (step 20000)

CPU-only two-scale inference, validation and core measurement of the E3 network (`F:\astro_essais\starRemoval\train\E3\snap\step_20000.pt`), used for the E3 rows of the README tables.

- `inferer_v2_cpu.py`: copy of `Portée M27 – 25 nov/inferer_v2.py` (two-scale V2 inference), CPU-only.
- `valider_cpu.py`: copy of `Validation – 22 nov/valider_reseau.py`, CPU-only.
- `coeurs_cpu.py`: copy of `Tour N7 – 25 nov/coeurs_mesure.py` (cores column), CPU-only.
- `liste.py`: copy of the validation list; `chaine_e3.py` (inference then validation) and `val_e3.py` (validation alone) chain them.

Nothing else was changed in the copies. The scripts hard-code the paths of the author's machine (`C:\Users\...`, `F:\float_ia\...`): they are given for traceability and must be adapted before any run.
