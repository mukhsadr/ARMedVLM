$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$cases = @(
  "C:\Users\adams\Documents\Spleen\inputs\CT1.nii.gz",
  "C:\Users\adams\Documents\Spleen\inputs\CT2.nii.gz",
  "C:\Users\adams\Documents\Spleen\inputs\CT3.nii.gz",
  "C:\Users\adams\Documents\Spleen\inputs\CT4.nii.gz"
)
$single = Join-Path $root "scripts\run_preprocess.ps1"

foreach ($ct in $cases) {
  if (Test-Path $ct) {
    Write-Host ""
    Write-Host "Preprocessing $ct"
    & $single $ct
  }
}
