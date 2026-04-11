from __future__ import annotations

import argparse
import math
from pathlib import Path

import pyvista as pv
import SimpleITK as sitk
from pyvistaqt import QtInteractor
from PySide6 import QtCore, QtWidgets


def read_image(path: str) -> sitk.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return sitk.ReadImage(str(p))


def sitk_to_grid(image: sitk.Image) -> pv.ImageData:
    arr = sitk.GetArrayFromImage(image).astype("float32")
    arr = arr.transpose(2, 1, 0)
    grid = pv.ImageData()
    grid.dimensions = arr.shape
    grid.spacing = image.GetSpacing()
    grid.origin = image.GetOrigin()
    grid.point_data["values"] = arr.ravel(order="F")
    return grid


def mask_to_mesh(mask_image: sitk.Image):
    grid = sitk_to_grid(mask_image)
    mesh = grid.contour(isosurfaces=[0.5], scalars="values")
    return mesh if mesh.n_points > 0 else None


class TimelineWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        ct_paths: list[str],
        mask_paths: list[str],
        labels: list[str] | None = None,
        spin_axis: str = "z",
        spin_degrees: float = 360.0,
        spin_frames: int = 240,
        transition_frames: int = 45,
        hold_frames: int = 20,
        fps: int = 30,
        elevation_deg: float = 8.0,
        zoom_factor: float = 1.0,
        center_on_mask: bool = False,
        loop_playback: bool = False,
    ):
        super().__init__()
        self.setWindowTitle("Registered CT Timeline")
        self.resize(1500, 920)
        self.ct_paths = ct_paths
        self.mask_paths = mask_paths
        self.labels = labels or [Path(p).stem for p in ct_paths]
        self.index = 0
        self.playing = False
        self.spin_phase = 0
        self.spin_steps = max(2, spin_frames)
        self.transition_steps = max(1, transition_frames)
        self.hold_frames = max(0, hold_frames)
        self.fps = max(1, fps)
        self.spin_axis = spin_axis.lower()
        self.spin_degrees = spin_degrees
        self.elevation_deg = elevation_deg
        self.zoom_factor = zoom_factor
        self.center_on_mask = center_on_mask
        self.loop_playback = loop_playback
        self.phase = "spin"
        self.phase_counter = 0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("Prev")
        self.play_btn = QtWidgets.QPushButton("Play")
        self.next_btn = QtWidgets.QPushButton("Next")
        self.label = QtWidgets.QLabel("Frame 1")
        self.prev_btn.clicked.connect(self.prev_frame)
        self.play_btn.clicked.connect(self.toggle_play)
        self.next_btn.clicked.connect(self.next_frame)
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.next_btn)
        controls.addWidget(self.label)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.plotter = QtInteractor(central)
        layout.addWidget(self.plotter.interactor, 1)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer_interval_ms = max(15, int(1000 / self.fps))
        self.volume_actor = None
        self.mask_actor = None
        self.object_center = (0.0, 0.0, 0.0)
        self.render_current(reset=True)

    def render_current(self, reset: bool = False) -> None:
        ct_img = read_image(self.ct_paths[self.index])
        ct_grid = sitk_to_grid(ct_img)
        self.plotter.clear()
        self.volume_actor = self.plotter.add_volume(
            ct_grid,
            scalars="values",
            cmap="bone",
            opacity=[0.0, 0.0, 0.06, 0.18, 0.34, 0.58, 0.82, 1.0],
            clim=(-250, 1800),
            shade=True,
            ambient=0.34,
            diffuse=0.92,
            specular=0.18,
            blending="composite",
        )
        self.object_center = tuple(float(v) for v in ct_grid.center)
        try:
            self.volume_actor.origin = self.object_center
            self.volume_actor.position = (0.0, 0.0, 0.0)
            self.volume_actor.orientation = (0.0, 0.0, 0.0)
        except Exception:
            pass
        if self.mask_paths[self.index]:
            mesh = mask_to_mesh(read_image(self.mask_paths[self.index]))
            if mesh is not None:
                self.mask_actor = self.plotter.add_mesh(mesh, color="#9b5cff", opacity=0.7, smooth_shading=True)
                try:
                    if self.center_on_mask:
                        self.object_center = tuple(float(v) for v in mesh.center)
                    self.mask_actor.origin = self.object_center
                    self.mask_actor.position = (0.0, 0.0, 0.0)
                    self.mask_actor.orientation = (0.0, 0.0, 0.0)
                    self.volume_actor.origin = self.object_center
                except Exception:
                    pass
            else:
                self.mask_actor = None
        else:
            self.mask_actor = None
        self.plotter.set_background("#20262b")
        if reset:
            self.plotter.reset_camera()
        try:
            self.plotter.camera.focal_point = self.object_center
            if self.elevation_deg:
                self.plotter.camera.elevation(self.elevation_deg)
                self.plotter.camera.orthogonalize_view_up()
            if self.zoom_factor and self.zoom_factor != 1.0:
                self.plotter.camera.zoom(self.zoom_factor)
        except Exception:
            pass
        name = self.labels[self.index] if self.index < len(self.labels) else Path(self.ct_paths[self.index]).name
        frame_title = (
            f"Spleen over time | Frame {self.index + 1}/{len(self.ct_paths)} | {name}\n"
            "Registered to CT4 fixed space"
        )
        self.label.setText(frame_title.replace("\n", "   "))
        self.plotter.add_text(frame_title, position="upper_left", font_size=12, color="white", name="timeline_hud")
        self.plotter.render()

    def prev_frame(self) -> None:
        self.index = (self.index - 1) % len(self.ct_paths)
        self.phase = "spin"
        self.phase_counter = 0
        self.render_current()

    def next_frame(self) -> None:
        if self.index == len(self.ct_paths) - 1 and not self.loop_playback:
            self.playing = False
            self.timer.stop()
            self.play_btn.setText("Play")
            return
        self.index = (self.index + 1) % len(self.ct_paths)
        self.phase = "spin"
        self.phase_counter = 0
        self.render_current()

    def _tick(self) -> None:
        if self.volume_actor is None:
            return
        self.phase_counter += 1

        if self.phase == "spin":
            t = self.phase_counter / self.spin_steps
            angle = math.sin(t * math.pi) * (self.spin_degrees / 2.0)
        elif self.phase == "hold":
            angle = 0.0
        else:
            angle = 0.0

        orientation = (0.0, 0.0, 0.0)
        if self.spin_axis == "x":
            orientation = (angle, 0.0, 0.0)
        elif self.spin_axis == "y":
            orientation = (0.0, angle, 0.0)
        else:
            orientation = (0.0, 0.0, angle)

        self.volume_actor.orientation = orientation
        if self.mask_actor is not None:
            self.mask_actor.orientation = orientation
        try:
            self.plotter.camera.focal_point = self.object_center
        except Exception:
            pass
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

        if self.phase == "spin" and self.phase_counter >= self.spin_steps:
            self.phase = "hold"
            self.phase_counter = 0
        elif self.phase == "hold" and self.phase_counter >= self.hold_frames:
            self.next_frame()

    def toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing:
            self.timer.start(self.timer_interval_ms)
            self.play_btn.setText("Pause")
        else:
            self.timer.stop()
            self.play_btn.setText("Play")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cts", nargs="+", required=True)
    parser.add_argument("--masks", nargs="*", default=[])
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--spin-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--spin-degrees", type=float, default=360.0)
    parser.add_argument("--spin-frames", type=int, default=240)
    parser.add_argument("--transition-frames", type=int, default=45)
    parser.add_argument("--hold-frames", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--elevation-deg", type=float, default=8.0)
    parser.add_argument("--zoom-factor", type=float, default=1.0)
    parser.add_argument("--center-on-mask", action="store_true")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    masks = list(args.masks)
    while len(masks) < len(args.cts):
        masks.append("")
    labels = list(args.labels)
    while len(labels) < len(args.cts):
        labels.append(Path(args.cts[len(labels)]).stem)

    app = QtWidgets.QApplication([])
    window = TimelineWindow(
        args.cts,
        masks,
        labels=labels,
        spin_axis=args.spin_axis,
        spin_degrees=args.spin_degrees,
        spin_frames=args.spin_frames,
        transition_frames=args.transition_frames,
        hold_frames=args.hold_frames,
        fps=args.fps,
        elevation_deg=args.elevation_deg,
        zoom_factor=args.zoom_factor,
        center_on_mask=args.center_on_mask,
        loop_playback=args.loop,
    )
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
