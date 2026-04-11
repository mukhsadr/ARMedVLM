$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$viewer = Join-Path $root "app\viewer.py"

$ctPath = if ($args.Count -gt 0) { $args[0] } else { "C:\Users\adams\Documents\Spleen\inputs\CT2_1mm.nii.gz" }
$maskPath = if ($args.Count -gt 1) { $args[1] } else { "" }
$maskDir = "C:\Users\adams\Documents\Spleen\outputs\deepspleenseg\masks"
$preDir = Join-Path $root "outputs\preprocessed"

if ($ctPath) { $ctPath = $ctPath.Trim() }
if ($maskPath) { $maskPath = $maskPath.Trim() }

$stem = [System.IO.Path]::GetFileNameWithoutExtension([System.IO.Path]::GetFileNameWithoutExtension($ctPath))
$cleanCandidate = Join-Path $preDir "${stem}_cleaned_ct.nii.gz"
$spleenCandidate = Join-Path $preDir "${stem}_spleen_mask.nii.gz"

if (Test-Path $cleanCandidate) {
  $ctPath = $cleanCandidate
}

if (-not (Test-Path $py)) {
  throw "Environment missing. Run .\\scripts\\setup_windows.ps1 first."
}

if (-not (Test-Path $ctPath)) {
  throw "CT file not found: $ctPath"
}

if (-not $maskPath) {
  if (Test-Path $spleenCandidate) {
    $maskPath = $spleenCandidate
  } else {
    $candidate = Join-Path $maskDir "case_${stem}_mask.nii.gz"
    if (Test-Path $candidate) {
      $maskPath = $candidate
    }
  }
}

Write-Host "CT:   $ctPath"
if ($maskPath -and -not ($maskPath.Contains("`n") -or $maskPath.Contains("`r")) -and (Test-Path $maskPath)) {
  Write-Host "Mask: $maskPath"
} else {
  Write-Host "Mask: none"
  Write-Host "Known masks:"
  Get-ChildItem $maskDir -Filter "*.nii.gz" | ForEach-Object { Write-Host "  $($_.FullName)" }
  $maskPath = ""
}

& $py $viewer --ct $ctPath $(if ($maskPath) { @("--mask", $maskPath) })
