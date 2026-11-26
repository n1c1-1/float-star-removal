# Download deeper Gaia extract

- Date : 2026-11-20 à 19:44 (heure de Paris)
- Numéro du jour : 04
- Type d'agent : general-purpose
- Texte : consigne telle que donnée à l'agent, après nettoyage (voir prompts/LISEZMOI.md)

---

Task: download a DEEPER Gaia DR3 star catalogue extract (the owner authorised the download) for the 6 test images of the classical star-removal work, so the script can remove faint stars down to G ≈ 20.5–21. The current offline extract (<CACHE_FLOAT>/cache/gaia_extracts, read only, never modify it) stops around G 13–17.

Steps:
1. Find how FLOAT queries and stores Gaia: <FLOAT>\astro_gpu (search for gaia_archive, gaia_extracts, VizieR, GAVO; read only, never modify). Reuse its query code/format (columns: position, G, BP, RP, and whatever else the extracts store) so the new files are drop-in compatible; if its format cannot go deeper, store a parallel format with the same columns.
2. Footprints: from <SCRIPTS>\Retrait étoiles par modèle – 20 nov\sorties\<image>\astrometrie.npz and astrometrie.txt (read only) — the 6 images: rosette, ngc6543, m57, horsehead, ngc869 (set_38), 58_hoo. Cone or box covering each full image + 5 % margin.
3. Query Gaia DR3 (gaia_source) down to G ≤ 21 for each footprint (VizieR I/355/gaiadr3 or the GAVO/ESA TAP service FLOAT already uses). Split large/dense fields into tiles if the service limits rows; be polite (sequential requests, retries with backoff). Send no personal information in requests (generic client name).
4. Write ONLY inside <SCRIPTS>\Catalogue Gaia profond – 20 nov\ (create it with a French LISEZMOI.txt: what, which work, 2026-11-20, created by session astro-f1, source service and query, magnitude limit, deletable). Nothing anywhere else (not FLOAT's cache, not F:, no temp leftovers). PYTHONDONTWRITEBYTECODE=1. Python: python. CPU only. Stop only your own processes; never PID 73024.
Report (French, short): per image the number of stars and the faintest G reached, compared with the current extract's count/limit on the same footprint; file paths and format; any service limit hit. Do not touch the star-removal scripts.
