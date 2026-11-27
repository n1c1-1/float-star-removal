# _lancer.ps1
# Role     : Exact launch line of the chain orchestrator (lancer_chaine_tuiles.py, budget 3 h, 4 threads).
# Stage    : 15. Classical baseline versus networks on 512 px tiles (26-27 Nov) (see scripts/README.md)
# Inputs   : lancer_borne.ps1 (scripts/); the paths below are those of the author's machine
# Outputs  : lanceur.log / inference.log in F:\float_ia\Article – 26 nov\baseline\
# Example  : _lancer.ps1
$a = '-I "C:\Users\Nicolas\sandbox\starRemoval\Article – 26 nov\baseline\lancer_chaine_tuiles.py" --budget-h 3'
& 'C:\Users\Nicolas\sandbox\starRemoval\lancer_borne.ps1' -Fils 4 -Log 'F:\float_ia\Article – 26 nov\baseline\lanceur.log' -Exe 'C:\Users\Nicolas\sandbox\astro\.venv\Scripts\python.exe' -Args $a -Dossier 'C:\Users\Nicolas\sandbox\starRemoval\Article – 26 nov\baseline'
