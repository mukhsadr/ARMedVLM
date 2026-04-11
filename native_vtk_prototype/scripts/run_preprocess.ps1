$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "app\ct_preprocess.py"
$ctPath = if ($args.Count -gt 0) { $args[0] } else { "C:\Users\adams\Documents\Spleen\inputs\CT4.nii.gz" }
$maskPath = if ($args.Count -gt 1) { $args[1] } else { "" }
$outDir = Join-Path $root "outputs\preprocessed"

if (-not (Test-Path $py)) {
  throw "Environment missing. Run .\\scripts\\setup_windows.ps1 first."
}
if (-not (Test-Path $ctPath)) {
  throw "CT not found: $ctPath"
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
& $py $script --ct $ctPath --out-dir $outDir $(if ($maskPath) { @("--mask", $maskPath) })
