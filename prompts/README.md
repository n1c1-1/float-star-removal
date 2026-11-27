# The instructions given to the agents

**Note on language.** The prompts themselves stay in their original language (a French/English mix), as a **verbatim historical record**. They are not translated and must not be edited. Only this index is in English.

## The role of each actor

This work was conducted by **Nicolas** with an artificial-intelligence assistant (Claude, Anthropic) as the **main session**: it decides, splits the work, judges the results by eye and reports to Nicolas. It does not itself launch the long computations: it entrusts each step to an **agent** (another instance of the assistant) to whom it gives a **written instruction**. The agent executes (writes a script, launches a computation, measures, looks at sheets), then returns a short report. The main session reads the report, looks at the images, and decides what comes next. The substantive decisions (which data, which criteria, when to stop) come from Nicolas; the agents do not decide.

These instructions are published **as they were given**, after cleaning, so that one can see exactly what was asked, with which rules.

## Scope: the published lineage only

Only the instructions of the **lineage of the published models** are published: the classical Gaia-guided removal chain (20-24 Nov), the v33/v33g pairs (Nicolas's images and web images), the trainings N1 to N9 from E3, the blends, two-scale inference, the measurements and controls. The instructions for other work (other networks, other sets of pairs, the application's interface) are not part of the published method and are not here. The instructions that launched the E2 and E3 trainings (7-9 Nov, synthetic data only) were not found among the agent calls of the transcripts: these steps were driven directly in the main session; their scripts are in `scripts/`.

## Contents

`YYYY-MM-DD/NN_HHMM_title.md`: one instruction per file, grouped by day, numbered in the order of the day. The header gives the title (the description of the agent call, in English or in French), the Paris date and time, the agent type.

| day | instructions | subject |
|---|---|---|
| 2026-11-20 | 7 | first classical chain, tracks 2 and 3, deep Gaia catalogue, first judges |
| 2026-11-21 | 36 | versions v1 to v18 of the classical chain (one change per instruction) |
| 2026-11-22 | 13 | test images, synthetic star generator, validation batch, first training on real pairs |
| 2026-11-23 | 9 | chain fixes, tile choice, delinearisation, production v33 |
| 2026-11-24 | 5 | judge v33c/v33d, 2 h trial, web images |
| 2026-11-25 | 8 | large halos (N3), diagnosis, N5/N6, M31 core, N7, slow restoration |
| 2026-11-26 | 8 | blends, N8, N8L, N9, diagnostics, control set |

Total: **86 instructions**, from 20 to 26 November 2026.

## What was removed or replaced

- names of third-party software: "[third-party tool]"; instructions whose sole purpose concerned such software are not published;
- the short names used in the original instructions for the network ("réseau étoiles" and variants) designate **our** network, never a third-party tool;
- e-mail addresses, forum pseudonyms, personal accounts and repositories, names of people other than Nicolas, author credits of the web images;
- absolute paths: `<SCRIPTS>`, `<DONNEES>`, `<ESSAIS>`, `<FLOAT>`, `<DISQUE>`;
- commit co-signature lines;
- lines that referred to other datasets outside the lineage: "[passage retiré]".

The substance of each instruction (goal, rules, steps, stopping criteria) is unchanged.

## How to read them

The instructions almost always give: the goal and the reason ("POURQUOI"), the strict rules (where to write, which resources, what not to touch), the steps, and the expected report format. They also show the rule that structured the work: *one change at a time, and write beforehand "already tried" and "what differs"* (`docs/LESSONS.md`). The word "Fable" designates a "cold step-back" agent that is asked for an independent opinion after several trials without progress.

## Markers inside the prompts

Square-bracket markers such as `[outil de comparaison]` or `[logiciel tiers]` replace the names of third-party software and of one earlier network of the host application, removed before publication; `<HOME>` and `<FLOAT>` replace local paths. Early prompts mention a visual comparison with such a third-party tool: that comparison was only ever looked at and was never an input of any training set (see `docs/PROVENANCE.md` and `provenance/STATIC_ANALYSIS.md`). The prompts are otherwise unedited.
