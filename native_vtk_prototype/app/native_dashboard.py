from __future__ import annotations

import argparse
import collections
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyvista as pv
import requests
import SimpleITK as sitk
from pyvistaqt import QtInteractor
from PySide6 import QtCore, QtGui, QtWidgets

MP_HANDS = None
MP_DRAWING = None
CVZONE_DETECTOR = None
try:
    from cvzone.HandTrackingModule import HandDetector
    CVZONE_DETECTOR = HandDetector(
        staticMode=False,
        maxHands=1,
        modelComplexity=0,
        detectionCon=0.5,
        minTrackCon=0.5,
    )
except Exception:
    CVZONE_DETECTOR = None

try:
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        MP_HANDS = mp.solutions.hands
        MP_DRAWING = mp.solutions.drawing_utils
except Exception:
    MP_HANDS = None
    MP_DRAWING = None


def read_image(path: str) -> sitk.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return sitk.ReadImage(str(p))


def sitk_to_pyvista_grid(image: sitk.Image) -> tuple[pv.ImageData, np.ndarray]:
    arr_zyx = sitk.GetArrayFromImage(image).astype(np.float32)
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    grid = pv.ImageData()
    grid.dimensions = arr_xyz.shape
    grid.spacing = image.GetSpacing()
    grid.origin = image.GetOrigin()
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


def spleen_metrics(ct_arr: np.ndarray, mask_arr: np.ndarray | None, spacing: tuple[float, float, float]) -> dict[str, float] | None:
    if mask_arr is None:
      return None
    voxels = mask_arr > 0
    if not np.any(voxels):
        return None
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
    values = ct_arr[voxels]
    return {
        "voxels": int(voxels.sum()),
        "volume_ml": float(voxels.sum() * voxel_volume_mm3 / 1000.0),
        "mean_hu": float(values.mean()),
        "std_hu": float(values.std()),
    }


def classify_hand(landmarks: list) -> tuple[str, int, str, str, str]:
    finger_count = 0
    tip_pip = [(8, 6), (12, 10), (16, 14), (20, 18)]
    for tip, pip in tip_pip:
        if landmarks[tip].y < landmarks[pip].y:
            finger_count += 1
    if landmarks[4].x < landmarks[3].x:
        finger_count += 1

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    cx = float(sum(xs) / len(xs))
    cy = float(sum(ys) / len(ys))

    zone_x = "center"
    if cx < 0.38:
        zone_x = "left"
    elif cx > 0.62:
        zone_x = "right"

    zone_y = "middle"
    if cy < 0.24:
        zone_y = "top"
    elif cy > 0.80:
        zone_y = "bottom"

    pose = "other"
    if finger_count >= 4:
        pose = "open_palm"
    elif finger_count == 0:
        pose = "fist"
    elif finger_count == 2:
        pose = "v_sign"
    handedness = "unknown"
    return pose, finger_count, zone_x, zone_y, handedness


def classify_cvzone_hand(hand: dict, detector: HandDetector, frame_shape: tuple[int, int, int]) -> tuple[str, int, str, str, str]:
    fingers = detector.fingersUp(hand)
    finger_count = int(sum(fingers))
    cx, cy = hand["center"]
    frame_h, frame_w = frame_shape[:2]
    nx = float(cx / frame_w)
    ny = float(cy / frame_h)

    zone_x = "center"
    if nx < 0.38:
        zone_x = "left"
    elif nx > 0.62:
        zone_x = "right"

    zone_y = "middle"
    if ny < 0.24:
        zone_y = "top"
    elif ny > 0.80:
        zone_y = "bottom"

    pose = "other"
    if finger_count >= 4:
        pose = "open_palm"
    elif finger_count == 0:
        pose = "fist"
    elif finger_count == 2:
        pose = "v_sign"
    handedness = str(hand.get("type", "unknown")).lower()
    return pose, finger_count, zone_x, zone_y, handedness


def draw_dashed_line(frame: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], color: tuple[int, int, int], thickness: int = 1, dash: int = 10) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length <= 0:
        return
    for i in range(0, length, dash * 2):
        start = i / length
        end = min(i + dash, length) / length
        sx = int(x1 + (x2 - x1) * start)
        sy = int(y1 + (y2 - y1) * start)
        ex = int(x1 + (x2 - x1) * end)
        ey = int(y1 + (y2 - y1) * end)
        cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)


def draw_control_grid(frame: np.ndarray, pose: str, zone_x: str, zone_y: str, action: str) -> None:
    h, w = frame.shape[:2]
    x1 = w // 3
    x2 = (2 * w) // 3
    y1 = int(h * 0.18)
    y2 = int(h * 0.82)
    green = (90, 255, 140)

    labels = [
        ((w // 6) - 34, (y1 + y2) // 2 - 10, "PALM"),
        ((w // 6) - 38, (y1 + y2) // 2 + 14, "SPIN LEFT"),
        ((w // 2) - 54, (y1 + y2) // 2 - 10, "PALM ZOOM+"),
        ((w // 2) - 46, (y1 + y2) // 2 + 14, "FIST ZOOM-"),
        (((5 * w) // 6) - 34, (y1 + y2) // 2 - 10, "PALM"),
        (((5 * w) // 6) - 42, (y1 + y2) // 2 + 14, "SPIN RIGHT"),
    ]
    for x, y, text in labels:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, green, 1, cv2.LINE_AA)

    boxes = {
        "spin_left": ((0, y1), (x1, y2)),
        "zoom_in": ((x1, y1), (x2, y2)),
        "zoom_out": ((x1, y1), (x2, y2)),
        "spin_right": ((x2, y1), (w, y2)),
    }
    for key, ((ax1, ay1), (ax2, ay2)) in boxes.items():
        color = (120, 255, 170) if action == key else green
        thickness = 2 if action == key else 1
        cv2.rectangle(frame, (ax1 + 2, ay1 + 2), (ax2 - 2, ay2 - 2), color, thickness, cv2.LINE_AA)

    status = f"pose={pose} zone={zone_x}/{zone_y} action={action}"
    cv2.rectangle(frame, (12, 12), (420, 42), (10, 20, 25), -1)
    cv2.putText(frame, status, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


class AskWorker(QtCore.QObject):
    finished = QtCore.Signal(str)

    def __init__(self, metrics: dict | None, question: str, context: dict | None = None):
        super().__init__()
        self.metrics = metrics
        self.question = question
        self.context = context or {}

    @QtCore.Slot()
    def run(self) -> None:
        context_lines = [
            f"Current case: {self.context.get('case_label', 'unknown')}",
            f"Case role: {self.context.get('case_role', 'unknown')}",
            f"Mask availability: {self.context.get('mask_status', 'unknown')}",
            f"Software stack: {self.context.get('stack', 'SimpleITK preprocessing and registration, PyVista/VTK rendering, Ollama chat')}",
            f"Preprocessing: {self.context.get('preprocess', 'body mask cleanup, outside-body removal, cropping')}",
            f"Registration: {self.context.get('registration', 'none')}",
            f"Available timeline cases: {self.context.get('timeline', 'unknown')}",
        ]
        metrics_line = "Metrics: mask not available."
        if self.metrics:
            metrics_line = (
                f"Metrics: volume_ml={self.metrics['volume_ml']:.1f}, "
                f"mean_hu={self.metrics['mean_hu']:.1f}, "
                f"std_hu={self.metrics['std_hu']:.1f}, "
                f"voxels={self.metrics['voxels']}."
            )
        prompt = (
            "You are a medical imaging assistant for a spleen CT dashboard. "
            "Use only the provided preprocessing, registration, case, and mask information. "
            "Do not diagnose. Keep the answer concise and practical. "
            "Write in chatbot-style prose, not bullet points, numbered lists, or JSON. "
            "Start with spleen volume if available. Then say whether the measured spleen volume looks within a typical adult range or enlarged-looking based only on the provided volume, not a diagnosis. "
            "Then mention the preprocessing and registration context briefly. "
            "If something is unavailable, say so plainly.\n\n"
            + "\n".join(context_lines)
            + "\n"
            + metrics_line
            + f"\nQuestion: {self.question}"
        )
        try:
            res = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={"model": "llama3.2:3b", "prompt": prompt, "stream": False},
                timeout=120,
            )
            res.raise_for_status()
            text = res.json().get("response", "").strip()
            self.finished.emit(text or "No response from Ollama.")
        except Exception:
            if self.metrics:
                self.finished.emit(
                    f"Case {self.context.get('case_label', 'unknown')}. "
                    f"Registration: {self.context.get('registration', 'none')}. "
                    f"Spleen mask available. Volume {self.metrics['volume_ml']:.1f} mL. "
                    f"Mean HU {self.metrics['mean_hu']:.1f}."
                )
            else:
                self.finished.emit(
                    f"Case {self.context.get('case_label', 'unknown')}. "
                    f"Registration: {self.context.get('registration', 'none')}. "
                    "No spleen mask metrics available for this case."
                )


class MainWindow(QtWidgets.QMainWindow):
    gesture_signal = QtCore.Signal(str, int, str, str, str)

    def __init__(self, ct_path: str, mask_path: str = ""):
        super().__init__()
        self.setWindowTitle("SpleenVTK_AR Native Dashboard")
        self.resize(1600, 980)

        self.ct_path = ct_path
        self.mask_path = mask_path
        self.series_cts, self.series_masks, self.series_labels = self._discover_series(ct_path, mask_path)
        self.series_index = 0

        self.ct_img = None
        self.ct_grid = None
        self.ct_arr = None
        self.mask_img = None
        self.mask_arr = None
        self.mask_mesh = None
        self.metrics = None
        self.llm_context: dict[str, str] = {}
        self.hand_tracking_available = CVZONE_DETECTOR is not None or MP_HANDS is not None
        self.object_center = (0.0, 0.0, 0.0)

        self.current_preset = "bone"
        self.show_mask = self.mask_mesh is not None
        self.last_action_ts = 0.0
        self.capture = None
        self.hands = None
        self.cvzone_detector = None
        self.worker_thread = None
        self.volume_actor = None
        self.mask_actor = None
        self.object_yaw = 0.0
        self.auto_spin_active = False
        self.pose_history = collections.deque()
        self.pose_window_seconds = 0.16
        self.required_majority = 0.52
        self.min_emit_interval = 0.06
        self.last_action_signature = "none"
        self.spin_timer = QtCore.QTimer(self)
        self.spin_timer.timeout.connect(self._spin_tick)
        self.spin_direction = 0
        self.spin_step_index = 0
        self.spin_total_steps = 999999
        self.spin_total_degrees = 5.0
        self.active_hold_action = "none"
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._spin_tick)

        self._build_ui()
        self._style_ui()
        self._load_case(self.series_index)
        self._redraw_volume(reset=True)
        self._start_camera()
        self.gesture_signal.connect(self._handle_gesture)

    def _discover_series(self, ct_path: str, mask_path: str) -> tuple[list[str], list[str], list[str]]:
        root = Path(__file__).resolve().parents[1]
        pre = root / "outputs" / "preprocessed"
        reg = root / "outputs" / "registered"
        labels = ["CT1", "CT3", "CT4"]
        cts: list[str] = []
        masks: list[str] = []
        out_labels: list[str] = []
        for label in labels:
            if label == "CT4":
                ct = pre / f"{label}_cleaned_ct.nii.gz"
                mk = pre / f"{label}_spleen_mask.nii.gz"
            else:
                ct = reg / f"{label}_cleaned_ct_to_CT4_cleaned_ct_registered.nii.gz"
                mk = reg / f"{label}_cleaned_ct_to_CT4_cleaned_ct_spleen_mask.nii.gz"
            if ct.exists():
                cts.append(str(ct))
                masks.append(str(mk) if mk.exists() else "")
                out_labels.append(label)
        if not cts:
            return [ct_path], [mask_path], [Path(ct_path).stem]
        return cts, masks, out_labels

    def _load_case(self, index: int) -> None:
        self.series_index = index % len(self.series_cts)
        self.ct_path = self.series_cts[self.series_index]
        self.mask_path = self.series_masks[self.series_index]
        self.ct_img = read_image(self.ct_path)
        self.ct_grid, self.ct_arr = sitk_to_pyvista_grid(self.ct_img)
        self.mask_img = read_image(self.mask_path) if self.mask_path else None
        self.mask_arr = sitk.GetArrayFromImage(self.mask_img).astype(np.uint8) if self.mask_img else None
        self.mask_mesh = build_spleen_mesh(self.mask_img) if self.mask_img else None
        self.metrics = spleen_metrics(self.ct_arr, self.mask_arr, self.ct_img.GetSpacing())
        self.show_mask = self.mask_mesh is not None
        self.object_center = tuple(float(v) for v in (self.mask_mesh.center if self.mask_mesh is not None else self.ct_grid.center))
        current_label = self.series_labels[self.series_index] if self.series_labels else Path(self.ct_path).stem
        registration = "Fixed reference CT4 cleaned space"
        if current_label != "CT4":
            registration = f"{current_label} rigidly registered to CT4 cleaned space"
        self.llm_context = {
            "case_label": current_label,
            "case_role": "fixed reference" if current_label == "CT4" else "moving case aligned to fixed CT4 reference",
            "mask_status": "spleen mask available" if self.mask_path else "no spleen mask available",
            "stack": "SimpleITK preprocessing and rigid registration, connected-component body cleanup, PyVista/VTK 3D rendering, Ollama language model",
            "preprocess": "SimpleITK body thresholding at soft-tissue range, connected-component cleanup, 2D hole filling, morphological opening and closing, outside-body removal to air, cropped cleaned CT, optional spleen-only CT and cropped spleen mask",
            "registration": registration,
            "timeline": " -> ".join(self.series_labels) if self.series_labels else current_label,
        }

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(18)

        self.left_panel = QtWidgets.QWidget()
        self.left_panel.setFixedWidth(360)
        left = QtWidgets.QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)
        self.left_panel.setLayout(left)
        root.addWidget(self.left_panel, 0)

        self.camera_card = QtWidgets.QFrame()
        self.camera_card.setObjectName("glass")
        self.camera_card.setFixedHeight(340)
        cam_layout = QtWidgets.QVBoxLayout(self.camera_card)
        cam_layout.setContentsMargins(14, 12, 14, 12)
        self.camera_title = QtWidgets.QLabel("MediaPipe Camera")
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setFixedHeight(210)
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label = QtWidgets.QLabel("Status: starting...")
        self.gesture_label = QtWidgets.QLabel("Gesture: none")
        self.fps_label = QtWidgets.QLabel("FPS: 0.0")
        cam_layout.addWidget(self.camera_title)
        cam_layout.addWidget(self.camera_label)
        cam_layout.addWidget(self.status_label)
        cam_layout.addWidget(self.gesture_label)
        cam_layout.addWidget(self.fps_label)
        left.addWidget(self.camera_card)

        self.chat_card = QtWidgets.QFrame()
        self.chat_card.setObjectName("glass")
        self.chat_card.setMinimumHeight(260)
        self.chat_card.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        chat_layout = QtWidgets.QVBoxLayout(self.chat_card)
        chat_layout.setContentsMargins(16, 14, 16, 16)
        chat_layout.setSpacing(10)
        chat_title = QtWidgets.QLabel("Ask Llama")
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 8, 0, 4)
        row.setSpacing(10)
        self.chat_input = QtWidgets.QLineEdit("Summarize preprocessing, registration, and the spleen findings.")
        self.ask_button = QtWidgets.QPushButton("Ask Llama")
        self.ask_button.clicked.connect(self._ask_llama)
        row.addWidget(self.chat_input, 1)
        row.addWidget(self.ask_button)
        self.chat_output = QtWidgets.QPlainTextEdit()
        self.chat_output.setReadOnly(True)
        self.chat_output.setPlainText("Ask about preprocessing, registration, CT changes, and spleen mask metrics.")
        self.chat_output.setMinimumHeight(180)
        self.chat_output.setViewportMargins(0, 0, 0, 6)
        chat_layout.addWidget(chat_title)
        chat_layout.addLayout(row)
        chat_layout.addWidget(self.chat_output, 1)
        left.addWidget(self.chat_card, 1)

        self.viewer_card = QtWidgets.QFrame()
        self.viewer_card.setObjectName("glass")
        self.viewer_card.setProperty("viewerCard", True)
        self.viewer_card.setMinimumWidth(980)
        viewer_layout = QtWidgets.QVBoxLayout(self.viewer_card)
        viewer_layout.setContentsMargins(14, 12, 14, 14)
        viewer_layout.setSpacing(10)
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(8)
        self.soft_btn = QtWidgets.QPushButton("Soft Tissue")
        self.bone_btn = QtWidgets.QPushButton("Bone")
        self.mask_btn = QtWidgets.QPushButton("Toggle Spleen")
        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.prev_btn = QtWidgets.QPushButton("Prev")
        self.play_btn = QtWidgets.QPushButton("Play")
        self.next_btn = QtWidgets.QPushButton("Next")
        self.soft_btn.clicked.connect(lambda: self._set_preset("soft"))
        self.bone_btn.clicked.connect(lambda: self._set_preset("bone"))
        self.mask_btn.clicked.connect(self._toggle_mask)
        self.reset_btn.clicked.connect(lambda: self._redraw_volume(reset=True))
        self.prev_btn.clicked.connect(self._prev_frame)
        self.play_btn.clicked.connect(self._toggle_playback)
        self.next_btn.clicked.connect(self._next_frame)
        for btn in (self.prev_btn, self.play_btn, self.next_btn, self.soft_btn, self.bone_btn, self.mask_btn, self.reset_btn):
            toolbar.addWidget(btn)
        toolbar.addStretch(1)
        viewer_layout.addLayout(toolbar)

        self.plotter = QtInteractor(self.viewer_card)
        self.plotter.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.plotter.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.plotter.interactor.setStyleSheet("background: transparent; border: 0;")
        try:
            self.plotter.render_window.SetAlphaBitPlanes(1)
            self.plotter.renderer.SetUseFXAA(False)
            self.plotter.renderer.SetBackgroundAlpha(0.0)
        except Exception:
            pass
        viewer_layout.addWidget(self.plotter.interactor, 1)
        root.addWidget(self.viewer_card, 1)
        self._apply_glass_effects()

    def _apply_glass_effects(self) -> None:
        for widget in (self.camera_card, self.chat_card, self.viewer_card):
            shadow = QtWidgets.QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(28)
            shadow.setOffset(0, 10)
            shadow.setColor(QtGui.QColor(0, 0, 0, 95))
            widget.setGraphicsEffect(shadow)

    def _style_ui(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0f1720; color: #eef3f7; font-family: Segoe UI; }
            QFrame#glass {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255,255,255,34),
                    stop:0.12 rgba(30,48,66,168),
                    stop:1 rgba(10,18,28,152)
                );
                border: 1px solid rgba(255,255,255,40);
                border-radius: 16px;
            }
            QFrame#glass[viewerCard="true"] {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255,255,255,22),
                    stop:0.16 rgba(28,42,58,112),
                    stop:1 rgba(10,18,28,92)
                );
                border: 1px solid rgba(255,255,255,26);
            }
            QLabel { font-size: 13px; }
            QLabel[role="title"] {
                font-size: 14px;
                font-weight: 700;
                color: #dff8fb;
                letter-spacing: 0.3px;
            }
            QPushButton {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255,255,255,26),
                    stop:1 rgba(92,225,230,38)
                );
                border: 1px solid rgba(92, 225, 230, 90);
                border-radius: 10px;
                padding: 9px 12px;
                min-height: 18px;
            }
            QPushButton:hover { background: rgba(92, 225, 230, 55); }
            QLineEdit, QPlainTextEdit {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255,255,255,14),
                    stop:1 rgba(0,0,0,64)
                );
                border: 1px solid rgba(255,255,255,34);
                border-radius: 10px;
                padding: 10px 12px;
            }
            QPlainTextEdit {
                margin-bottom: 8px;
            }
            QLineEdit {
                min-height: 22px;
            }
        """)
        for label in (self.camera_title,):
            label.setProperty("role", "title")
        self.left_panel.setStyleSheet("background: transparent;")

    def _add_volume(self) -> None:
        cmap = "bone" if self.current_preset == "bone" else "copper"
        if self.current_preset == "bone":
            opacity = [0.0, 0.0, 0.05, 0.14, 0.30, 0.56, 0.82, 1.0]
            clim = (-250, 1800)
        else:
            opacity = [0.0, 0.0, 0.02, 0.05, 0.10, 0.18, 0.28, 0.40]
            clim = (-110, 220)

        self.volume_actor = self.plotter.add_volume(
            self.ct_grid,
            scalars="values",
            cmap=cmap,
            opacity=opacity,
            clim=clim,
            shade=True,
            ambient=0.34,
            diffuse=0.92,
            specular=0.18,
            blending="composite",
        )
        try:
            self.volume_actor.origin = self.object_center
        except Exception:
            pass

    def _redraw_volume(self, reset: bool = False) -> None:
        self.plotter.clear()
        self._add_volume()
        self.mask_actor = None
        if self.show_mask and self.mask_mesh is not None:
            self.mask_actor = self.plotter.add_mesh(self.mask_mesh, color="#41d6c3", opacity=0.72, smooth_shading=True)
            try:
                self.mask_actor.origin = self.object_center
            except Exception:
                pass
        info = [
            f"CT: {self.series_labels[self.series_index] if self.series_labels else Path(self.ct_path).name}",
            f"Preset: {'Bone' if self.current_preset == 'bone' else 'Soft Tissue'}",
            f"Mask: {'on' if self.show_mask and self.mask_mesh is not None else 'off'}",
            "Camera zones: left palm=spin left | center palm/fist=zoom in/out | right palm=spin right",
        ]
        if not self.hand_tracking_available:
            info.append("Hand tracking unavailable in this mediapipe build; use mouse or toolbar.")
        if self.metrics:
            info.append(f"Spleen volume: {self.metrics['volume_ml']:.1f} mL")
        self.plotter.add_text("\n".join(info), position="upper_left", font_size=10, color="white")
        # Restore the older solid viewer background so the 3D object reads clearly.
        self.plotter.set_background("#5d666d")
        if reset:
            self.object_yaw = 0.0
            self.plotter.reset_camera()
            try:
                self.plotter.camera.focal_point = self.object_center
            except Exception:
                pass
        self._apply_actor_orientation()
        self.plotter.render()

    def _set_preset(self, preset: str) -> None:
        self.current_preset = preset
        self._redraw_volume()

    def _toggle_mask(self) -> None:
        if self.mask_mesh is None:
            return
        self.show_mask = not self.show_mask
        self._redraw_volume()

    def _prev_frame(self) -> None:
        self._load_case(self.series_index - 1)
        self._redraw_volume(reset=True)

    def _next_frame(self) -> None:
        self._load_case(self.series_index + 1)
        self._redraw_volume(reset=True)

    def _toggle_playback(self) -> None:
        if self.auto_spin_active:
            self.auto_spin_active = False
            self.play_timer.stop()
            self.play_btn.setText("Play")
        else:
            self.auto_spin_active = True
            self.play_timer.start(24)
            self.play_btn.setText("Pause")

    def _apply_gesture(self, gesture: str) -> None:
        camera = self.plotter.camera
        if gesture == "zoom_in":
            camera.zoom(1.025)
        elif gesture == "zoom_out":
            camera.zoom(0.975)
        elif gesture == "spin_left":
            self._start_spin(-1)
            return
        elif gesture == "spin_right":
            self._start_spin(1)
            return
        elif gesture == "reset_views":
            self.object_yaw = 0.0
            self.plotter.reset_camera()
        self._apply_actor_orientation()
        try:
            camera.focal_point = self.object_center
        except Exception:
            pass
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    def _apply_actor_orientation(self) -> None:
        orientation = (0.0, 0.0, float(self.object_yaw))
        if self.volume_actor is not None:
            self.volume_actor.orientation = orientation
        if self.mask_actor is not None:
            self.mask_actor.orientation = orientation

    def _start_spin(self, direction: int) -> None:
        self.spin_direction = direction
        self.spin_step_index = 0
        if not self.spin_timer.isActive():
            self.spin_timer.start(24)

    def _stop_spin(self) -> None:
        self.spin_timer.stop()
        self.spin_direction = 0
        self.spin_step_index = 0

    def _spin_tick(self) -> None:
        if self.volume_actor is None:
            self.spin_timer.stop()
            return
        if not self.auto_spin_active and self.spin_direction == 0:
            self.spin_timer.stop()
            return
        self.spin_step_index += 1
        if self.auto_spin_active:
            self.object_yaw = (self.object_yaw + 2.2) % 360.0
        else:
            step = self.spin_total_degrees
            self.object_yaw = (self.object_yaw + (step * self.spin_direction)) % 360.0
        self._apply_actor_orientation()
        try:
            self.plotter.camera.focal_point = self.object_center
        except Exception:
            pass
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    def _stable_pose(self, pose: str) -> tuple[str, bool]:
        now = time.time()
        self.pose_history.append((now, pose))
        while self.pose_history and now - self.pose_history[0][0] > self.pose_window_seconds:
            self.pose_history.popleft()

        if not self.pose_history:
            return "none", False

        counts: dict[str, int] = {}
        for _, p in self.pose_history:
            counts[p] = counts.get(p, 0) + 1
        dominant_pose = max(counts, key=counts.get)
        ratio = counts[dominant_pose] / len(self.pose_history)
        return dominant_pose, ratio >= self.required_majority

    @QtCore.Slot(str, int, str, str, str)
    def _handle_gesture(self, pose: str, finger_count: int, zone_x: str, zone_y: str, handedness: str) -> None:
        now = time.time()
        stable_pose, stable_enough = self._stable_pose(pose)
        action = "none"
        signature = f"{stable_pose}:{zone_x}"

        if stable_enough:
            if stable_pose == "open_palm":
                if zone_x == "left" and handedness == "left":
                    action = "spin_left"
                elif zone_x == "right" and handedness == "right":
                    action = "spin_right"
                else:
                    action = "zoom_in"
            elif stable_pose == "fist":
                if zone_x == "center":
                    action = "zoom_out"
            elif stable_pose == "v_sign":
                action = "reset_views"

        self.gesture_label.setText(
            f"Gesture: {stable_pose} ({finger_count}) hand={handedness} zone={zone_x} action={action} stable={stable_enough}"
        )

        if action in ("spin_left", "spin_right"):
            desired = -1 if action == "spin_left" else 1
            self.active_hold_action = action
            if self.spin_direction != desired:
                self._start_spin(desired)
            return

        if self.active_hold_action.startswith("spin"):
            self._stop_spin()
            self.active_hold_action = "none"

        if action == "reset_views":
            if signature != self.last_action_signature and now - self.last_action_ts >= self.min_emit_interval:
                self.last_action_ts = now
                self.last_action_signature = signature
                self._apply_gesture(action)
            return

        if action in ("zoom_in", "zoom_out"):
            if now - self.last_action_ts >= self.min_emit_interval:
                self.last_action_ts = now
                self._apply_gesture(action)
            self.last_action_signature = signature
            return

        self.last_action_signature = "none"

    def _start_camera(self) -> None:
        self.capture = cv2.VideoCapture(0)
        if CVZONE_DETECTOR is not None:
            self.cvzone_detector = CVZONE_DETECTOR
            self.hands = None
        elif self.hand_tracking_available:
            self.hands = MP_HANDS.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.hands = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_camera)
        self.timer.start(40)
        self._last_fps_ts = time.time()
        self._fps_counter = 0

    def _update_camera(self) -> None:
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.status_label.setText("Status: camera read failed")
            return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb) if self.hands is not None else None

        pose, finger_count, zone_x, zone_y, handedness = "none", 0, "center", "middle", "unknown"
        preview_action = "none"
        if self.cvzone_detector is not None:
            hands, frame = self.cvzone_detector.findHands(frame, draw=True, flipType=False)
            if hands:
                pose, finger_count, zone_x, zone_y, handedness = classify_cvzone_hand(hands[0], self.cvzone_detector, frame.shape)
                self.gesture_signal.emit(pose, finger_count, zone_x, zone_y, handedness)
            else:
                self.gesture_signal.emit("none", 0, "center", "middle", "unknown")
        elif res is not None and res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            MP_DRAWING.draw_landmarks(frame, hand, MP_HANDS.HAND_CONNECTIONS)
            pose, finger_count, zone_x, zone_y, handedness = classify_hand(hand.landmark)
            self.gesture_signal.emit(pose, finger_count, zone_x, zone_y, handedness)
        else:
            self.gesture_signal.emit("none", 0, "center", "middle", "unknown")

        if pose == "open_palm":
            if zone_x == "left" and handedness == "left":
                preview_action = "spin_left"
            elif zone_x == "right" and handedness == "right":
                preview_action = "spin_right"
            elif zone_x == "center":
                preview_action = "zoom_in"
        elif pose == "fist":
            if zone_x == "center":
                preview_action = "zoom_out"
        elif pose == "v_sign":
            preview_action = "reset"

        draw_control_grid(frame, pose, zone_x, zone_y, preview_action)

        self._fps_counter += 1
        now = time.time()
        if now - self._last_fps_ts >= 1.0:
            fps = self._fps_counter / (now - self._last_fps_ts)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self._fps_counter = 0
            self._last_fps_ts = now

        if self.cvzone_detector is not None:
            self.status_label.setText(
                "Status: camera active | cvzone hand tracking | left=rotate center=zoom in right=rotate | fist=zoom out"
            )
        elif self.hands is None:
            self.status_label.setText("Status: camera active | hand tracking unavailable in current mediapipe build")
        else:
            self.status_label.setText(
                f"Status: camera active | left=rotate center=zoom in right=rotate | fist=zoom out"
            )
        rgb_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_show.shape
        qimg = QtGui.QImage(rgb_show.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.camera_label.width(),
            self.camera_label.height(),
            QtCore.Qt.KeepAspectRatioByExpanding,
            QtCore.Qt.SmoothTransformation,
        )
        self.camera_label.setPixmap(pix)

    def _ask_llama(self) -> None:
        question = self.chat_input.text().strip()
        if not question:
            return
        self.ask_button.setEnabled(False)
        self.chat_output.setPlainText("Thinking...")
        self.worker_thread = QtCore.QThread(self)
        self.worker = AskWorker(self.metrics, question, self.llm_context)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._finish_llama)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    @QtCore.Slot(str)
    def _finish_llama(self, text: str) -> None:
        self.ask_button.setEnabled(True)
        self.chat_output.setPlainText(self._format_chat_text(text))

    def _format_chat_text(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^\s*[-*•]+\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.capture is not None:
            self.capture.release()
        if self.hands is not None:
            self.hands.close()
        return super().closeEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", required=True)
    parser.add_argument("--mask", default="")
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    window = MainWindow(args.ct, args.mask)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
