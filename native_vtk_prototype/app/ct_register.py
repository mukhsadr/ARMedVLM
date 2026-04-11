from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk


def read_image(path: str) -> sitk.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return sitk.ReadImage(str(p))


def euler_rigid_registration(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.15, seed=42)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Stage 1: correlation to find a good global alignment.
    reg.SetMetricAsCorrelation()
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-3,
        numberOfIterations=120,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-6,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial, inPlace=False)
    tx_corr = reg.Execute(fixed, moving)

    # Stage 2: mean squares refinement in the same CT modality.
    reg2 = sitk.ImageRegistrationMethod()
    reg2.SetInterpolator(sitk.sitkLinear)
    reg2.SetMetricSamplingStrategy(reg2.RANDOM)
    reg2.SetMetricSamplingPercentage(0.2, seed=43)
    reg2.SetShrinkFactorsPerLevel([2, 1])
    reg2.SetSmoothingSigmasPerLevel([1, 0])
    reg2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg2.SetMetricAsMeanSquares()
    reg2.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=100,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-6,
    )
    reg2.SetOptimizerScalesFromPhysicalShift()
    reg2.SetInitialTransform(tx_corr, inPlace=False)
    return reg2.Execute(fixed, moving)


def resample_to_fixed(fixed: sitk.Image, moving: sitk.Image, transform: sitk.Transform, is_label: bool = False) -> sitk.Image:
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    default = 0 if is_label else -1024.0
    out_type = sitk.sitkUInt8 if is_label else sitk.sitkFloat32
    return sitk.Resample(moving, fixed, transform, interp, default, out_type)


def register_case(
    fixed_ct_path: str,
    moving_ct_path: str,
    out_dir: str,
    moving_mask_path: str = "",
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fixed = read_image(fixed_ct_path)
    moving = read_image(moving_ct_path)

    fixed_name = Path(fixed_ct_path).name.replace(".nii.gz", "")
    moving_name = Path(moving_ct_path).name.replace(".nii.gz", "")

    tx = euler_rigid_registration(fixed, moving)
    registered = resample_to_fixed(fixed, moving, tx, is_label=False)

    out_ct = out / f"{moving_name}_to_{fixed_name}_registered.nii.gz"
    sitk.WriteImage(registered, str(out_ct))

    result = {
        "registered_ct": str(out_ct),
    }

    if moving_mask_path and Path(moving_mask_path).exists():
        moving_mask = read_image(moving_mask_path)
        registered_mask = resample_to_fixed(fixed, moving_mask, tx, is_label=True)
        out_mask = out / f"{moving_name}_to_{fixed_name}_spleen_mask.nii.gz"
        sitk.WriteImage(registered_mask, str(out_mask))
        result["registered_mask"] = str(out_mask)

    out_tx = out / f"{moving_name}_to_{fixed_name}_rigid.h5"
    try:
        if isinstance(tx, sitk.CompositeTransform):
            tx.FlattenTransform()
    except Exception:
        pass
    sitk.WriteTransform(tx, str(out_tx))
    result["transform"] = str(out_tx)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--moving-mask", default="")
    args = parser.parse_args()
    result = register_case(args.fixed, args.moving, args.out_dir, args.moving_mask)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
