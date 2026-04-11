$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "app\ct_register.py"
$preDir = Join-Path $root "outputs\preprocessed"
$regDir = Join-Path $root "outputs\registered"

$fixedRaw = if ($args.Count -gt 0) { $args[0] } else { "C:\Users\adams\Documents\Spleen\inputs\CT4.nii.gz" }
$fixedStem = [System.IO.Path]::GetFileNameWithoutExtension([System.IO.Path]::GetFileNameWithoutExtension($fixedRaw))
$fixed = Join-Path $preDir "${fixedStem}_cleaned_ct.nii.gz"
if (-not (Test-Path $fixed)) { $fixed = $fixedRaw }

$cases = @(
  "C:\Users\adams\Documents\Spleen\inputs\CT1.nii.gz",
  "C:\Users\adams\Documents\Spleen\inputs\CT3.nii.gz",
  "C:\Users\adams\Documents\Spleen\inputs\CT4.nii.gz"
)

New-Item -ItemType Directory -Force -Path $regDir | Out-Null

foreach ($raw in $cases) {
  if (-not (Test-Path $raw)) { continue }
  $stem = [System.IO.Path]::GetFileNameWithoutExtension([System.IO.Path]::GetFileNameWithoutExtension($raw))
  $moving = Join-Path $preDir "${stem}_cleaned_ct.nii.gz"
  if (-not (Test-Path $moving)) { $moving = $raw }
  if ($moving -eq $fixed) { continue }
  $movingMask = Join-Path $preDir "${stem}_spleen_mask.nii.gz"
  if (-not (Test-Path $movingMask)) { $movingMask = "" }
  Write-Host ""
  Write-Host "Registering $moving -> $fixed"
  & $py $script --fixed $fixed --moving $moving --out-dir $regDir $(if ($movingMask) { @("--moving-mask", $movingMask) })
}
