from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk


def read_image(path: str) -> sitk.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return sitk.ReadImage(str(p))


def largest_component(mask: sitk.Image) -> sitk.Image:
    cc = sitk.ConnectedComponent(mask)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    return sitk.Cast(relabeled == 1, sitk.sitkUInt8)


def fill_holes_2d(mask: sitk.Image) -> sitk.Image:
    filler = sitk.BinaryFillholeImageFilter()
    size = list(mask.GetSize())
    filled = sitk.Image(size, sitk.sitkUInt8)
    filled.CopyInformation(mask)
    for z in range(mask.GetSize()[2]):
        sl = mask[:, :, z]
        sl_filled = sitk.Cast(filler.Execute(sl), sitk.sitkUInt8)
        filled[:, :, z] = sl_filled
    return filled


def copy_geometry(dst: sitk.Image, src: sitk.Image) -> sitk.Image:
    dst.CopyInformation(src)
    return dst


def body_mask_from_ct(ct: sitk.Image) -> sitk.Image:
    body = sitk.BinaryThreshold(ct, lowerThreshold=-350, upperThreshold=3000, insideValue=1, outsideValue=0)
    body = sitk.BinaryMorphologicalClosing(body, [7, 7, 1])
    body = largest_component(body)
    body = fill_holes_2d(body)
    body = sitk.BinaryMorphologicalOpening(body, [3, 3, 1])
    body = sitk.BinaryMorphologicalClosing(body, [5, 5, 1])
    return sitk.Cast(body, sitk.sitkUInt8)


def bbox_from_mask(mask: sitk.Image) -> tuple[list[int], list[int]] | None:
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    if not stats.HasLabel(1):
        return None
    x, y, z, sx, sy, sz = stats.GetBoundingBox(1)
    return [int(x), int(y), int(z)], [int(sx), int(sy), int(sz)]


def apply_roi(image: sitk.Image, roi: list[int], size: list[int]) -> sitk.Image:
    return sitk.RegionOfInterest(image, size=size, index=roi)


def make_cleaned_ct(ct: sitk.Image, body_mask: sitk.Image) -> sitk.Image:
    outside = sitk.Image(ct.GetSize(), sitk.sitkFloat32)
    outside.CopyInformation(ct)
    outside = sitk.Add(outside, -1024.0)
    ct_f = sitk.Cast(ct, sitk.sitkFloat32)
    body_f = sitk.Cast(body_mask, sitk.sitkFloat32)
    return ct_f * body_f + outside * sitk.Cast(sitk.InvertIntensity(body_mask, maximum=1), sitk.sitkFloat32)


def make_spleen_only_ct(cleaned_ct: sitk.Image, spleen_mask: sitk.Image | None) -> sitk.Image | None:
    if spleen_mask is None:
        return None
    spleen_mask = sitk.Cast(spleen_mask > 0, sitk.sitkUInt8)
    outside = sitk.Image(cleaned_ct.GetSize(), sitk.sitkFloat32)
    outside.CopyInformation(cleaned_ct)
    outside = sitk.Add(outside, -1024.0)
    cleaned_f = sitk.Cast(cleaned_ct, sitk.sitkFloat32)
    return cleaned_f * sitk.Cast(spleen_mask, sitk.sitkFloat32) + outside * sitk.Cast(
        sitk.InvertIntensity(spleen_mask, maximum=1), sitk.sitkFloat32
    )


def resolve_default_mask(ct_path: Path) -> Path | None:
    stem = ct_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif ct_path.suffix:
        stem = ct_path.stem
    mask_dir = Path(r"C:\Users\adams\Documents\Spleen\outputs\deepspleenseg\masks")
    candidate = mask_dir / f"case_{stem}_mask.nii.gz"
    return candidate if candidate.exists() else None


def preprocess_case(ct_path: str, out_dir: str, mask_path: str = "") -> dict[str, str]:
    ct_p = Path(ct_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem = ct_p.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif ct_p.suffix:
        stem = ct_p.stem

    ct = read_image(str(ct_p))
    body_mask = body_mask_from_ct(ct)
    cleaned_ct = make_cleaned_ct(ct, body_mask)
    bbox = bbox_from_mask(body_mask)
    if bbox is not None:
        roi, size = bbox
        cleaned_ct = apply_roi(cleaned_ct, roi, size)
        cropped_body_mask = apply_roi(body_mask, roi, size)
    else:
        cropped_body_mask = body_mask

    spleen_mask = None
    resolved_mask = Path(mask_path) if mask_path else resolve_default_mask(ct_p)
    if resolved_mask and resolved_mask.exists():
        spleen_mask = read_image(str(resolved_mask))
        spleen_mask = sitk.Resample(
            spleen_mask,
            ct,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        if bbox is not None:
            spleen_mask = apply_roi(spleen_mask, roi, size)

    body_path = out / f"{stem}_body_mask.nii.gz"
    clean_path = out / f"{stem}_cleaned_ct.nii.gz"
    sitk.WriteImage(copy_geometry(cropped_body_mask, cleaned_ct), str(body_path))
    sitk.WriteImage(cleaned_ct, str(clean_path))

    result = {
        "cleaned_ct": str(clean_path),
        "body_mask": str(body_path),
    }

    if spleen_mask is not None:
        spleen_crop_path = out / f"{stem}_spleen_mask.nii.gz"
        spleen_only = make_spleen_only_ct(cleaned_ct, spleen_mask)
        spleen_only_path = out / f"{stem}_spleen_only_ct.nii.gz"
        sitk.WriteImage(copy_geometry(spleen_mask, cleaned_ct), str(spleen_crop_path))
        sitk.WriteImage(spleen_only, str(spleen_only_path))
        result["spleen_mask"] = str(spleen_crop_path)
        result["spleen_only_ct"] = str(spleen_only_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mask", default="")
    args = parser.parse_args()
    result = preprocess_case(args.ct, args.out_dir, args.mask)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
