# ======================================================================
# suivre_inference.py
# Role     : Runs the CPU network inference on each tile as soon as the classical chain has finished it.
# Stage    : 15. Classical baseline versus networks on 512 px tiles (26-27 Nov) (see scripts/README.md)
# Inputs   : PID of the chain launcher; classique_<image>.npy files as they appear
# Outputs  : candidats\<model>_<image>.npy, inference.log
# Example  : python -I suivre_inference.py <launcher pid>   (launch line in _lancer2.ps1)
# ======================================================================
r"""CPU inference of the networks (E3 N2 N5 N7 N7N5) on the tiles as the classical chain finishes them (4 threads, in parallel with the 4 threads
of the chain launcher: 8 threads in all). Stops when the launcher process (given PID) no longer exists and everything is done.
usage: suivre_inference.py <launcher pid>"""
import sys, time
sys.path.insert(0, r"C:\Users\Nicolas\sandbox\starRemoval\Article – 26 nov\baseline")
import psutil
import mesurer_tuiles as M

pid = int(sys.argv[1])
while True:
    vivant = psutil.pid_exists(pid)
    fait = [n for n in M.noms_tuiles() if M.candidat_path("classique", n).exists()]
    todo = [n for n in fait if any(not M.candidat_path(s, n).exists() for s in M.MODELES)]
    if todo:
        M.log(f"a faire : {todo}")
        M.inferer(list(M.MODELES), todo)
    elif not vivant:
        break
    else:
        time.sleep(60)
M.log("inference terminee")
