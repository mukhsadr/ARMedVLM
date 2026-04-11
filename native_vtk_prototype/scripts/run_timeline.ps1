$ErrorActionPreference = "Stop"

# ------------------------------------------------------------
# Launch longitudinal viewer with smooth Z-axis spin + transitions
# - Spins around the body using Z as the rotation axis
# - Keeps rotation centered on the volume / spleen region
# - Adds smoother timepoint transitions
# ------------------------------------------------------------

$root   = Split-Path -Parent $PSScriptRoot
$py     = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "app\timeline_viewer.py"

$regDir = Join-Path $root "outputs\registered"
$preDir = Join-Path $root "outputs\preprocessed"

# ------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------
$fixedStem = if ($args.Count -gt 0) { $args[0] } else { "CT4" }

# Ordered timeline to show
$orderedCases = @("CT1", "CT3", "CT4")

# Camera / animation settings
$fps               = 30
$spinDegrees       = 360
$spinFrames        = 240      # higher = smoother spin
$transitionFrames  = 45       # smooth blend between timepoints
$holdFrames        = 20       # pause briefly on each phase
$elevationDeg      = 8        # small tilt so 3D shape reads better
$zoomFactor        = 1.0      # 1.0 = default distance
$spinAxis          = "z"      # rotate around z-axis
$loopPlayback      = $true

# ------------------------------------------------------------
# BUILD INPUT SERIES
# ------------------------------------------------------------
$fixedCt = Join-Path $preDir "${fixedStem}_cleaned_ct.nii.gz"

$cts = New-Object System.Collections.Generic.List[string]
$masks = New-Object System.Collections.Generic.List[string]
$labels = New-Object System.Collections.Generic.List[string]

foreach ($case in $orderedCases) {
    if ($case -eq $fixedStem) {
        if (Test-Path $fixedCt) {
            $cts.Add($fixedCt)

            $fixedMask = Join-Path $preDir "${fixedStem}_spleen_mask.nii.gz"
            if (Test-Path $fixedMask) {
                $masks.Add($fixedMask)
            } else {
                $masks.Add("")
            }

            $labels.Add($case)
        }
    }
    else {
        $regCt = Join-Path $regDir "${case}_cleaned_ct_to_${fixedStem}_cleaned_ct_registered.nii.gz"
        $regMask = Join-Path $regDir "${case}_cleaned_ct_to_${fixedStem}_cleaned_ct_spleen_mask.nii.gz"

        if (Test-Path $regCt) {
            $cts.Add($regCt)

            if (Test-Path $regMask) {
                $masks.Add($regMask)
            } else {
                $masks.Add("")
            }

            $labels.Add($case)
        }
    }
}

if ($cts.Count -eq 0) {
    throw "No registered series found. Run .\scripts\run_register_all.ps1 first."
}

Write-Host ""
Write-Host "Fixed reference : $fixedStem"
Write-Host "CT volumes      : $($cts.Count)"
Write-Host "Spin axis       : $spinAxis"
Write-Host "Spin frames     : $spinFrames"
Write-Host "Transition      : $transitionFrames frames"
Write-Host ""

# ------------------------------------------------------------
# BUILD COMMAND
# ------------------------------------------------------------
$cmd = @(
    $script,
    "--cts"
)

$cmd += $cts

$cmd += @(
    "--masks"
)

$cmd += $masks

$cmd += @(
    "--labels"
)

$cmd += $labels

$cmd += @(
    "--spin-axis", $spinAxis,
    "--spin-degrees", "$spinDegrees",
    "--spin-frames", "$spinFrames",
    "--transition-frames", "$transitionFrames",
    "--hold-frames", "$holdFrames",
    "--fps", "$fps",
    "--elevation-deg", "$elevationDeg",
    "--zoom-factor", "$zoomFactor",
    "--center-on-mask"
)

if ($loopPlayback) {
    $cmd += "--loop"
}

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
& $py @cmd