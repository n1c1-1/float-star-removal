# lancer_borne.ps1: launches a command at low priority, cores 4-15, fixed threads.
# Example: .\lancer_borne.ps1 -Fils 4 -Log chaine.out -Exe python -Args "-I script.py"
# Launches a bounded computation: low priority, cores 4-15 only (0-3 stay free for Windows and remote access),
# fixed threads. Child processes inherit the priority and the cores. Writes the PID in the log.
# Example: powershell -File lancer_borne.ps1 -Fils 4 -Log <DONNEES>\x.log -Exe python -Args "-I script.py --opt 1"
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
