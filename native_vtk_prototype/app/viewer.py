from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import SimpleITK as sitk


def read_image(path: str) -> sitk.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return sitk.ReadImage(str(p))


def sitk_to_pyvista_grid(image: sitk.Image) -> tuple[pv.ImageData, np.ndarray]:
    arr_zyx = sitk.GetArrayFromImage(image).astype(np.float32)
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    spacing = image.GetSpacing()
    origin = image.GetOrigin()

    grid = pv.ImageData()
    grid.dimensions = arr_xyz.shape
    grid.spacing = spacing
    grid.origin = origin
    grid.point_data["values"] = arr_xyz.ravel(order="F")
    return grid, arr_zyx


def build_spleen_mesh(mask_image: sitk.Image) -> pv.PolyData | None:
    mask_grid, mask_arr = sitk_to_pyvista_grid(mask_image)
    if np.count_nonzero(mask_arr) == 0:
        return None
    mesh = mask_grid.contour(isosurfaces=[0.5], scalars="values")
    if mesh.n_points == 0:
        return None
    return mesh.smooth(n_iter=20, relaxation_factor=0.08)


def spleen_metrics(ct_arr: np.ndarray, mask_arr: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, float] | None:
    voxels = mask_arr > 0
    if not np.any(voxels):
        return None
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
    values = ct_arr[voxels]
    return {
        "voxels": int(voxels.sum()),
        "volume_ml": voxels.sum() * voxel_volume_mm3 / 1000.0,
        "mean_hu": float(values.mean()),
        "std_hu": float(values.std()),
    }


def add_volume(plotter: pv.Plotter, grid: pv.ImageData, preset: str) -> None:
    plotter.clear_actors()
    cmap = "bone" if preset == "bone" else "gray"
    if preset == "bone":
      opacity = [0.0, 0.0, 0.06, 0.18, 0.34, 0.58, 0.82, 1.0]
      clim = (-250, 1800)
    else:
      opacity = [0.0, 0.0, 0.05, 0.14, 0.28, 0.42, 0.58, 0.74]
      clim = (-120, 260)

    plotter.add_volume(
        grid,
        scalars="values",
        cmap=cmap,
        opacity=opacity,
        clim=clim,
        shade=True,
        ambient=0.25,
        diffuse=0.8,
        specular=0.1,
        blending="composite",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", required=True)
    parser.add_argument("--mask", default="")
    args = parser.parse_args()

    ct_img = read_image(args.ct)
    ct_grid, ct_arr = sitk_to_pyvista_grid(ct_img)

    mask_img = read_image(args.mask) if args.mask else None
    mask_arr = sitk.GetArrayFromImage(mask_img).astype(np.uint8) if mask_img else None
    mask_mesh = build_spleen_mesh(mask_img) if mask_img else None

    metrics = spleen_metrics(ct_arr, mask_arr, ct_img.GetSpacing()) if mask_img is not None else None

    pv.set_plot_theme("document")
    plotter = pv.Plotter(window_size=(1400, 900))
    plotter.background_color = (0.04, 0.06, 0.08)

    state = {"preset": "soft", "show_mask": mask_mesh is not None, "object_mode": False}

    def redraw() -> None:
        source_grid = ct_grid
        if state["object_mode"] and mask_img is not None:
            spleen_grid, _ = sitk_to_pyvista_grid(mask_img)
            source_grid = spleen_grid
        add_volume(plotter, source_grid, state["preset"])
        if state["show_mask"] and mask_mesh is not None:
            plotter.add_mesh(
                mask_mesh,
                color="#9b5cff",
                opacity=0.65,
                smooth_shading=True,
                name="spleen_mask",
            )
        info_lines = [
            f"CT: {Path(args.ct).name}",
            f"Preset: {'Bone' if state['preset'] == 'bone' else 'Soft Tissue'}",
            f"Mask: {'on' if state['show_mask'] and mask_mesh is not None else 'off'}",
            f"Mode: {'Spleen Object' if state['object_mode'] else 'Body Context'}",
            "Keys: 1 soft, 2 bone, 3 mask toggle, 4 object mode, r reset",
        ]
        if metrics:
            info_lines.append(f"Spleen volume: {metrics['volume_ml']:.1f} mL")
            info_lines.append(f"Mean HU: {metrics['mean_hu']:.1f}  Std HU: {metrics['std_hu']:.1f}")
        plotter.add_text("\n".join(info_lines), position="upper_left", font_size=11, color="white", name="hud")
        plotter.reset_camera()
        plotter.render()

    def set_soft() -> None:
        state["preset"] = "soft"
        redraw()

    def set_bone() -> None:
        state["preset"] = "bone"
        redraw()

    def toggle_mask() -> None:
        if mask_mesh is None:
            return
        state["show_mask"] = not state["show_mask"]
        redraw()

    def toggle_object_mode() -> None:
        if mask_img is None:
            return
        state["object_mode"] = not state["object_mode"]
        redraw()

    def reset_camera() -> None:
        plotter.reset_camera()
        plotter.render()

    plotter.add_key_event("1", set_soft)
    plotter.add_key_event("2", set_bone)
    plotter.add_key_event("3", toggle_mask)
    plotter.add_key_event("4", toggle_object_mode)
    plotter.add_key_event("r", reset_camera)

    redraw()
    plotter.show(title="SpleenVTK_AR Viewer")


if __name__ == "__main__":
    main()
