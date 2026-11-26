# lancer_borne.ps1 : lance une commande a priorite basse, coeurs 4-15, fils fixes.
# Exemple : .\lancer_borne.ps1 -Fils 4 -Log chaine.out -Exe python -Args "-I script.py"
# Lance un calcul borné : priorité basse, cœurs 4-15 seulement (0-3 restent libres pour Windows et l'accès à distance),
# fils fixés. Les processus enfants héritent de la priorité et des cœurs. Écrit le PID dans le journal.
# Exemple : powershell -File lancer_borne.ps1 -Fils 4 -Log <DONNEES>\x.log -Exe python -Args "-I script.py --opt 1"
param(
  [Parameter(Mandatory=$true)][string]$Args,
  [string]$Exe = "python",
  [int]$Fils = 4,
  [Parameter(Mandatory=$true)][string]$Log,
  [string]$Dossier = (Get-Location).Path
)
foreach ($v in "OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS","NUMEXPR_NUM_THREADS","TORCH_NUM_THREADS") {
  Set-Item -Path "env:$v" -Value "$Fils"
}
$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES = "0"   # 4060 Ti seulement (à vérifier par nvidia-smi : la 3070 ne doit jamais servir)
$p = Start-Process -FilePath $Exe -ArgumentList $Args -WorkingDirectory $Dossier -PassThru -WindowStyle Hidden `
     -RedirectStandardOutput $Log -RedirectStandardError ($Log + ".err")
$p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
$p.ProcessorAffinity = [IntPtr]0xFFF0
"PID $($p.Id) fils=$Fils log=$Log"
