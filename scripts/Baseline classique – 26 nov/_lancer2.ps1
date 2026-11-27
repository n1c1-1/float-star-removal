# _lancer2.ps1
# Role     : Exact launch line of the inference follower (suivre_inference.py <launcher pid>, 4 threads).
# Stage    : 15. Classical baseline versus networks on 512 px tiles (26-27 Nov) (see scripts/README.md)
# Inputs   : lancer_borne.ps1 (scripts/); the paths below are those of the author's machine
# Outputs  : lanceur.log / inference.log in F:\float_ia\Article – 26 nov\baseline\
# Example  : _lancer2.ps1
$a = '-I "C:\Users\Nicolas\sandbox\starRemoval\Article – 26 nov\baseline\suivre_inference.py" 68792'
& 'C:\Users\Nicolas\sandbox\starRemoval\lancer_borne.ps1' -Fils 4 -Log 'F:\float_ia\Article – 26 nov\baseline\inference.log' -Exe 'C:\Users\Nicolas\sandbox\astro\.venv\Scripts\python.exe' -Args $a -Dossier 'C:\Users\Nicolas\sandbox\starRemoval\Article – 26 nov\baseline'
