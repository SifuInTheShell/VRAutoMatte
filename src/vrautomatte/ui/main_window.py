"""Main application window for VRAutoMatte."""

from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from loguru import logger

from vrautomatte.pipeline.runner import (
    OutputFormat,
    PipelineConfig,
    PipelineProgress,
    ProjectionType,
)
from vrautomatte.ui.preview import PreviewWidget
from vrautomatte.ui.themes import (
    DARK_COLORS,
    DARK_STYLE,
    LIGHT_COLORS,
    LIGHT_STYLE,
)
from vrautomatte.ui.worker import InstallWorker, PipelineWorker
from vrautomatte.utils.ffmpeg import check_ffmpeg, get_video_info
from vrautomatte.utils.gpu import get_device_info
from vrautomatte.utils.masks import ensure_mask, get_mask_path
from vrautomatte.utils.settings import load_settings, save_settings





def _check_matanyone2() -> bool:
    """Check if MatAnyone 2 + SAM2 dependencies are installed."""
    try:
        import sam2  # noqa: F401
        import matanyone2  # noqa: F401
        return True
    except ImportError:
        return False


class MainWindow(QMainWindow):
    """Main application window.

    Layout:
    - File I/O section (input file, output file)
    - Settings section (model, quality, projection, FOV)
    - Preview section (source frame | generated matte)
    - Batch queue (optional, collapsible)
    - Action bar (Start button, progress bar, status)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VRAutoMatte")
        self.setMinimumSize(860, 720)
        self.worker: PipelineWorker | None = None
        self._batch_queue: list[dict] = []
        self._batch_index = 0
        self._settings = load_settings()
        self._video_info: dict | None = None
        self._is_dark = self._settings.get("dark_theme", False)
        self._colors = DARK_COLORS if self._is_dark else LIGHT_COLORS
        self._setup_ui()
        self._restore_settings()
        self._update_device_label()
        self._apply_theme()
        self.setAcceptDrops(True)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(18, 18, 18, 18)

        # ── File I/O ──
        io_group = QGroupBox("Files")
        io_layout = QVBoxLayout(io_group)

        # Input row
        in_row = QHBoxLayout()
        in_row.addWidget(QLabel("Input:"))
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText(
            "Select input video file..."
        )
        self.input_edit.setReadOnly(True)
        in_row.addWidget(self.input_edit, stretch=1)
        self.input_btn = QPushButton("Browse...")
        self.input_btn.clicked.connect(self._browse_input)
        in_row.addWidget(self.input_btn)
        self.add_batch_btn = QPushButton("+ Queue")
        self.add_batch_btn.setObjectName("addBatchButton")
        self.add_batch_btn.setToolTip(
            "Add current file to the batch queue"
        )
        self.add_batch_btn.clicked.connect(self._add_to_batch)
        in_row.addWidget(self.add_batch_btn)
        io_layout.addLayout(in_row)

        # Output row
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText(
            "Output file path (auto-generated if empty)..."
        )
        out_row.addWidget(self.output_edit, stretch=1)
        self.output_btn = QPushButton("Browse...")
        self.output_btn.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_btn)
        io_layout.addLayout(out_row)

        # Video info
        self.info_label = QLabel("")
        io_layout.addWidget(self.info_label)

        # Frame range (optional)
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Frame range:"))
        self.start_frame_edit = QLineEdit()
        self.start_frame_edit.setPlaceholderText("Start")
        self.start_frame_edit.setFixedWidth(80)
        self.start_frame_edit.setToolTip(
            "First frame to process (1-based). "
            "Leave empty for start of video."
        )
        range_row.addWidget(self.start_frame_edit)
        range_row.addWidget(QLabel("–"))
        self.end_frame_edit = QLineEdit()
        self.end_frame_edit.setPlaceholderText("End")
        self.end_frame_edit.setFixedWidth(80)
        self.end_frame_edit.setToolTip(
            "Last frame to process (inclusive). "
            "Leave empty for end of video."
        )
        range_row.addWidget(self.end_frame_edit)
        self.range_info = QLabel("")
        self.range_info.setStyleSheet(
            "font-size: 10px; font-style: italic;"
        )
        range_row.addWidget(self.range_info)
        range_row.addStretch()
        self.start_frame_edit.textChanged.connect(
            lambda: self._update_range_info()
        )
        self.end_frame_edit.textChanged.connect(
            lambda: self._update_range_info()
        )
        io_layout.addLayout(range_row)

        root.addWidget(io_group)

        # ── Settings ──
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Row 1: Model + Output Format
        row1 = QHBoxLayout()
        model_label = QLabel("Matting Model:")
        model_label.setToolTip(
            "AI model used for person segmentation.\n\n"
            "• mobilenetv3 — Fastest, good for previewing "
            "and lower-end GPUs (~50 fps at 1080p). "
            "Detects ALL people in the frame.\n"
            "• resnet50 — Better edge quality, slightly "
            "slower (~30 fps at 1080p). "
            "Detects ALL people. Best for crowds.\n"
            "• MatAnyone 2 — Sharpest edges, best "
            "hair/transparency (~8 fps, ~6 GB VRAM). "
            "Tracks ONE person from the first frame. "
            "Use for single-subject close-ups only."
        )
        row1.addWidget(model_label)
        self.model_combo = QComboBox()
        self._ma2_available = _check_matanyone2()
        self.model_combo.addItems([
            "mobilenetv3 — all people, fast",
            "resnet50 — all people, quality",
            "MatAnyone 2 — single subject"
            if self._ma2_available
            else "MatAnyone 2 — click to install",
        ])
        self.model_combo.setToolTip(model_label.toolTip())
        self.model_combo.currentIndexChanged.connect(
            self._on_model_changed
        )
        row1.addWidget(self.model_combo)
        row1.addSpacing(20)
        format_label = QLabel("Output Format:")
        format_label.setToolTip(
            "What the pipeline produces.\n\n"
            "• Matte Only — Grayscale alpha matte video. "
            "White = person, black = background. "
            "Use for compositing in video editors.\n"
            "• DeoVR Alpha Pack — Full passthrough pipeline "
            "for Meta Quest. Converts to fisheye, packs "
            "video + red-channel matte vertically. "
            "Output filename gets _ALPHA suffix."
        )
        row1.addWidget(format_label)
        self.format_combo = QComboBox()
        self.format_combo.setToolTip(format_label.toolTip())
        self.format_combo.addItems([
            "Matte Only", "DeoVR Alpha Pack"
        ])
        self.format_combo.currentIndexChanged.connect(
            self._on_format_changed
        )
        row1.addWidget(self.format_combo)
        row1.addStretch()
        settings_layout.addLayout(row1)

        # Row 2: Quality (CRF) slider + Downsample
        row2 = QHBoxLayout()
        crf_label = QLabel("Quality (CRF):")
        crf_label.setToolTip(
            "Constant Rate Factor — controls output video "
            "quality vs file size.\n\n"
            "• 10–14: Visually lossless, large files\n"
            "• 15–19: High quality, good balance (default: 18)\n"
            "• 20–25: Medium quality, smaller files\n"
            "• 26–30: Lower quality, smallest files\n\n"
            "Lower number = better quality = bigger file."
        )
        row2.addWidget(crf_label)
        self.crf_slider = QSlider(Qt.Orientation.Horizontal)
        self.crf_slider.setRange(10, 30)
        self.crf_slider.setValue(18)
        self.crf_slider.setTickInterval(2)
        self.crf_slider.setTickPosition(
            QSlider.TickPosition.TicksBelow
        )
        self.crf_slider.setToolTip(crf_label.toolTip())
        self.crf_slider.valueChanged.connect(self._update_crf_label)
        row2.addWidget(self.crf_slider)
        self.crf_label = QLabel("18")
        self.crf_label.setFixedWidth(30)
        row2.addWidget(self.crf_label)
        row2.addSpacing(20)
        ds_label = QLabel("Downsample:")
        ds_label.setToolTip(
            "Processing resolution for RVM models. "
            "Lower = faster but less precise edges.\n\n"
            "• 0.125 — Fastest, roughest edges. Good for "
            "quick previews.\n"
            "• 0.25 — Balanced speed and quality (default). "
            "Recommended for most use.\n"
            "• 0.5 — Higher quality edges, 2× slower.\n"
            "• 1.0 — Full resolution, best edges but "
            "slowest and most VRAM.\n\n"
            "Has no effect on MatAnyone 2 (always full res)."
        )
        row2.addWidget(ds_label)
        self.downsample_combo = QComboBox()
        self.downsample_combo.addItems([
            "0.125 (fastest)", "0.25 (balanced)",
            "0.5 (quality)", "1.0 (full res)",
        ])
        self.downsample_combo.setToolTip(ds_label.toolTip())
        self.downsample_combo.setCurrentIndex(1)
        row2.addWidget(self.downsample_combo)
        row2.addStretch()
        settings_layout.addLayout(row2)

        # Row 3: VR-specific (conditional)
        self.vr_row_widget = QWidget()
        vr_row = QHBoxLayout(self.vr_row_widget)
        vr_row.setContentsMargins(0, 0, 0, 0)
        proj_label = QLabel("Projection:")
        proj_label.setToolTip(
            "Input video projection type.\n\n"
            "• Equirectangular → Fisheye — Standard 360° "
            "VR video. Will be converted to fisheye for "
            "DeoVR passthrough.\n"
            "• Already Fisheye — Input is already in "
            "fisheye format. Skips conversion step."
        )
        vr_row.addWidget(proj_label)
        self.projection_combo = QComboBox()
        self.projection_combo.addItems([
            "Equirectangular → Fisheye", "Already Fisheye"
        ])
        self.projection_combo.setToolTip(
            proj_label.toolTip()
        )
        vr_row.addWidget(self.projection_combo)
        vr_row.addSpacing(20)
        fov_label_text = QLabel("Fisheye FOV:")
        fov_label_text.setToolTip(
            "Field of view for the fisheye projection.\n\n"
            "• 180° — Standard hemisphere\n"
            "• 190° — Slight over-capture (default for "
            "most VR cameras)\n"
            "• 200°+ — Ultra-wide, may distort edges\n\n"
            "Match this to your camera's actual FOV "
            "for best results."
        )
        vr_row.addWidget(fov_label_text)
        self.fov_slider = QSlider(Qt.Orientation.Horizontal)
        self.fov_slider.setRange(170, 210)
        self.fov_slider.setValue(180)
        self.fov_slider.setToolTip(
            fov_label_text.toolTip()
        )
        self.fov_slider.valueChanged.connect(self._update_fov_label)
        vr_row.addWidget(self.fov_slider)
        self.fov_label = QLabel("180°")
        self.fov_label.setFixedWidth(35)
        vr_row.addWidget(self.fov_label)
        vr_row.addSpacing(20)
        codec_label = QLabel("Codec:")
        codec_label.setToolTip(
            "Video codec for the output file.\n\n"
            "• HEVC (H.265) — Better compression, smaller "
            "files. Recommended for Quest. Requires "
            "hardware decoder support.\n"
            "• H.264 — Wider compatibility, slightly "
            "larger files. Use if HEVC playback fails."
        )
        vr_row.addWidget(codec_label)
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(["HEVC (H.265)", "H.264"])
        self.codec_combo.setToolTip(codec_label.toolTip())
        vr_row.addWidget(self.codec_combo)
        vr_row.addStretch()
        self.vr_row_widget.setVisible(False)
        settings_layout.addWidget(self.vr_row_widget)

        # Row 4: SBS stereo
        sbs_row = QHBoxLayout()
        self.sbs_check = QCheckBox(
            "SBS Stereo (per-eye matting)"
        )
        self.sbs_check.setToolTip(
            "Process each eye independently for side-by-side "
            "stereo VR videos.\n\n"
            "Auto-detected when aspect ratio ≥ 1.9:1 "
            "(e.g. 3840×1920).\n"
            "Each eye gets its own matting pass for better "
            "quality at stereo boundaries.\n"
            "Override manually if auto-detection is wrong."
        )
        sbs_row.addWidget(self.sbs_check)
        self.sbs_auto_label = QLabel("")
        sbs_row.addWidget(self.sbs_auto_label)
        sbs_row.addSpacing(20)
        self.pov_check = QCheckBox("POV Mode")
        self.pov_check.setToolTip(
            "For first-person VR content where the camera "
            "operator's body is visible.\n\n"
            "SAM2 detects the operator's body on the first "
            "frame and excludes it from the matte — only "
            "other people are kept.\n\n"
            "• MatAnyone 2: Best quality (instance-level "
            "matting, re-runs SAM2 on scene changes)\n"
            "• RVM: Fast mode (static mask subtraction, "
            "rougher but much faster)\n\n"
            "Requires SAM2 (installed with MatAnyone 2)."
        )
        self.pov_check.stateChanged.connect(
            self._update_pov_warning
        )
        sbs_row.addWidget(self.pov_check)
        self.pov_warning = QLabel("")
        sbs_row.addWidget(self.pov_warning)
        sbs_row.addStretch()
        settings_layout.addLayout(sbs_row)

        # Row 5: Temp directory
        temp_row = QHBoxLayout()
        temp_row.addWidget(QLabel("Temp directory:"))
        self.temp_dir_edit = QLineEdit()
        self.temp_dir_edit.setPlaceholderText(
            "System default"
        )
        self.temp_dir_edit.setReadOnly(True)
        self.temp_dir_edit.setToolTip(
            "Directory for temporary frame files during "
            "processing.\n\n"
            "Large videos (8K, 100k+ frames) can require "
            "hundreds of GB of temp space.\n"
            "Choose a fast drive with enough free space.\n\n"
            "Leave empty to use the system default temp "
            "directory."
        )
        temp_row.addWidget(self.temp_dir_edit, stretch=1)
        self.temp_browse_btn = QPushButton("Browse…")
        self.temp_browse_btn.clicked.connect(
            self._browse_temp_dir
        )
        temp_row.addWidget(self.temp_browse_btn)
        self.temp_clear_btn = QPushButton("Reset")
        self.temp_clear_btn.setToolTip(
            "Reset to system default temp directory"
        )
        self.temp_clear_btn.clicked.connect(
            lambda: self.temp_dir_edit.clear()
        )
        temp_row.addWidget(self.temp_clear_btn)
        settings_layout.addLayout(temp_row)

        # Row 6: Chunk size + Auto-resume
        chunk_row = QHBoxLayout()
        chunk_label = QLabel("Chunk size:")
        chunk_label.setToolTip(
            "Frames to extract per chunk.\n\n"
            "• 100 — Minimal disk usage, more overhead\n"
            "• 250 — Low disk, balanced\n"
            "• 500 — Default. Good balance\n"
            "• 1000 — Less overhead, more disk\n\n"
            "Smaller = less peak disk usage."
        )
        chunk_row.addWidget(chunk_label)
        self.chunk_size_combo = QComboBox()
        self.chunk_size_combo.addItems([
            "100", "250", "500 (default)", "1000"
        ])
        self.chunk_size_combo.setCurrentIndex(2)
        self.chunk_size_combo.setToolTip(
            chunk_label.toolTip()
        )
        chunk_row.addWidget(self.chunk_size_combo)
        chunk_row.addSpacing(20)
        self.resume_check = QCheckBox(
            "Auto-resume on restart"
        )
        self.resume_check.setChecked(True)
        self.resume_check.setToolTip(
            "Save progress after each chunk.\n"
            "If interrupted, resume from the last\n"
            "completed chunk on restart."
        )
        chunk_row.addWidget(self.resume_check)
        chunk_row.addStretch()
        settings_layout.addLayout(chunk_row)

        root.addWidget(settings_group)

        # ── Preview ──
        self.preview = PreviewWidget()
        self.preview.frame_scrubbed.connect(
            self._on_frame_scrubbed
        )
        root.addWidget(self.preview, stretch=1)

        # ── Batch Queue (collapsible) ──
        self.batch_group = QGroupBox(
            "Batch Queue (0 files)"
        )
        batch_layout = QVBoxLayout(self.batch_group)
        self.batch_list = QListWidget()
        self.batch_list.setMaximumHeight(100)
        self.batch_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        batch_layout.addWidget(self.batch_list)
        batch_btn_row = QHBoxLayout()
        self.batch_remove_btn = QPushButton("Remove Selected")
        self.batch_remove_btn.clicked.connect(
            self._remove_from_batch
        )
        batch_btn_row.addWidget(self.batch_remove_btn)
        self.batch_clear_btn = QPushButton("Clear All")
        self.batch_clear_btn.clicked.connect(self._clear_batch)
        batch_btn_row.addWidget(self.batch_clear_btn)
        batch_btn_row.addStretch()
        batch_layout.addLayout(batch_btn_row)
        self.batch_group.setVisible(False)
        root.addWidget(self.batch_group)

        # ── Action bar ──
        action_layout = QHBoxLayout()

        self.start_btn = QPushButton("▶  Start Processing")
        self.start_btn.setObjectName("startButton")
        self.start_btn.clicked.connect(self._start_processing)
        action_layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("cancelButton")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel_processing)
        action_layout.addWidget(self.cancel_btn)

        action_layout.addSpacing(16)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        action_layout.addWidget(self.progress_bar, stretch=1)

        root.addLayout(action_layout)

        # Disk usage estimate
        self.disk_label = QLabel("")
        self.disk_label.setObjectName("diskLabel")
        self.disk_label.setStyleSheet(
            "font-size: 10px; font-style: italic;"
        )
        root.addWidget(self.disk_label)

        # Status bar
        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        self.device_label = QLabel("")
        self.device_label.setObjectName("deviceLabel")
        status_row.addWidget(self.device_label)
        status_row.addSpacing(8)
        self.theme_btn = QPushButton("🌙")
        self.theme_btn.setObjectName("themeToggle")
        self.theme_btn.setToolTip("Toggle light / dark theme")
        self.theme_btn.clicked.connect(self._toggle_theme)
        status_row.addWidget(self.theme_btn)
        root.addLayout(status_row)

        # Convert all plain-text tooltips to rich text so Qt
        # enables automatic word-wrapping instead of clipping.
        for w in self.findChildren(QWidget):
            tip = w.toolTip()
            if tip and "<" not in tip:
                w.setToolTip(tip.replace("\n", "<br>"))

    # ── Settings Persistence ──

    def _restore_settings(self):
        """Restore saved settings to the UI widgets."""
        s = self._settings
        self.model_combo.setCurrentIndex(
            s.get("model_variant", 0)
        )
        self.downsample_combo.setCurrentIndex(
            s.get("downsample_ratio", 1)
        )
        self.crf_slider.setValue(s.get("crf", 18))
        self.format_combo.setCurrentIndex(
            s.get("output_format", 0)
        )
        self.projection_combo.setCurrentIndex(
            s.get("projection", 0)
        )
        self.fov_slider.setValue(s.get("fisheye_fov", 180))
        self.codec_combo.setCurrentIndex(s.get("codec", 0))
        self.sbs_check.setChecked(s.get("is_sbs", False))
        self.pov_check.setChecked(s.get("pov_mode", False))
        self.temp_dir_edit.setText(s.get("temp_dir", ""))
        self.chunk_size_combo.setCurrentIndex(
            s.get("chunk_size", 2)
        )
        self.resume_check.setChecked(
            s.get("auto_resume", True)
        )
        self.resize(
            s.get("window_width", 900),
            s.get("window_height", 780),
        )

    def _save_current_settings(self):
        """Capture current UI state and persist to disk."""
        self._settings.update({
            "model_variant": self.model_combo.currentIndex(),
            "downsample_ratio": self.downsample_combo.currentIndex(),
            "crf": self.crf_slider.value(),
            "output_format": self.format_combo.currentIndex(),
            "projection": self.projection_combo.currentIndex(),
            "fisheye_fov": self.fov_slider.value(),
            "codec": self.codec_combo.currentIndex(),
            "is_sbs": self.sbs_check.isChecked(),
            "pov_mode": self.pov_check.isChecked(),
            "dark_theme": self._is_dark,
            "temp_dir": self.temp_dir_edit.text(),
            "chunk_size": self.chunk_size_combo.currentIndex(),
            "auto_resume": self.resume_check.isChecked(),
            "window_width": self.width(),
            "window_height": self.height(),
        })
        save_settings(self._settings)

    def closeEvent(self, event):
        """Save settings when the window is closed."""
        self._save_current_settings()
        super().closeEvent(event)

    # ── Drag & Drop ──

    _VIDEO_EXTENSIONS = {
        ".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv",
    }

    def dragEnterEvent(self, event):
        """Accept drag if it contains video file URLs."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            has_video = any(
                Path(u.toLocalFile()).suffix.lower()
                in self._VIDEO_EXTENSIONS
                for u in urls if u.isLocalFile()
            )
            if has_video:
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event):
        """Handle dropped files: single → input, multiple → batch."""
        urls = event.mimeData().urls()
        video_paths = [
            u.toLocalFile() for u in urls
            if u.isLocalFile()
            and Path(u.toLocalFile()).suffix.lower()
            in self._VIDEO_EXTENSIONS
        ]

        if not video_paths:
            return

        if len(video_paths) == 1:
            # Single file → set as current input
            path = video_paths[0]
            self.input_edit.setText(path)
            self._settings["last_input_dir"] = str(
                Path(path).parent
            )
            self._auto_output_name(path)
            self._show_video_info(path)
            logger.info(f"Dropped input: {path}")
        else:
            # Multiple files → add all to batch queue
            for path in video_paths:
                self._add_file_to_batch(path)
            logger.info(
                f"Dropped {len(video_paths)} files "
                "into batch queue"
            )

        event.acceptProposedAction()

    # ── Batch Queue ──

    def _add_to_batch(self):
        """Add the current input/output pair to the batch queue."""
        input_path = self.input_edit.text()
        output_path = self.output_edit.text()
        if not input_path:
            QMessageBox.warning(
                self, "No Input",
                "Select an input file first.",
            )
            return
        if not output_path:
            self._auto_output_name(input_path)
            output_path = self.output_edit.text()

        entry = {
            "input": input_path,
            "output": output_path,
        }
        self._batch_queue.append(entry)

        name = Path(input_path).name
        item = QListWidgetItem(f"{name}  →  {Path(output_path).name}")
        self.batch_list.addItem(item)
        self._update_batch_header()
        self.batch_group.setVisible(True)

        # Clear input for next file
        self.input_edit.clear()
        self.output_edit.clear()

    def _add_file_to_batch(self, input_path: str):
        """Add a file to the batch queue with auto output name.

        Args:
            input_path: Path to the video file.
        """
        p = Path(input_path)
        stem = p.stem
        suffix = p.suffix
        output_path = str(
            p.parent / f"{stem}_matte{suffix}"
        )

        entry = {
            "input": input_path,
            "output": output_path,
        }
        self._batch_queue.append(entry)

        item = QListWidgetItem(
            f"{p.name}  →  {Path(output_path).name}"
        )
        self.batch_list.addItem(item)
        self._update_batch_header()
        self.batch_group.setVisible(True)
        self.info_label.clear()

    def _remove_from_batch(self):
        """Remove selected items from the batch queue."""
        for item in reversed(self.batch_list.selectedItems()):
            idx = self.batch_list.row(item)
            self.batch_list.takeItem(idx)
            if idx < len(self._batch_queue):
                self._batch_queue.pop(idx)
        self._update_batch_header()
        if not self._batch_queue:
            self.batch_group.setVisible(False)

    def _clear_batch(self):
        """Clear the entire batch queue."""
        self._batch_queue.clear()
        self.batch_list.clear()
        self._update_batch_header()
        self.batch_group.setVisible(False)

    def _update_batch_header(self):
        n = len(self._batch_queue)
        self.batch_group.setTitle(f"Batch Queue ({n} file{'s' if n != 1 else ''})")

    # ── Slots ──

    def _browse_input(self):
        start_dir = self._settings.get("last_input_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Video",
            start_dir,
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm);;"
            "All Files (*)",
        )
        if path:
            self.input_edit.setText(path)
            self._settings["last_input_dir"] = str(
                Path(path).parent
            )
            self._auto_output_name(path)
            self._show_video_info(path)

    def _browse_output(self):
        start_dir = self._settings.get("last_output_dir", "")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output As",
            self.output_edit.text() or start_dir,
            "Video Files (*.mp4 *.mkv);;All Files (*)",
        )
        if path:
            self.output_edit.setText(path)
            self._settings["last_output_dir"] = str(
                Path(path).parent
            )

    def _browse_temp_dir(self):
        """Let user pick a custom temp directory."""
        start_dir = self.temp_dir_edit.text() or ""
        path = QFileDialog.getExistingDirectory(
            self, "Select Temp Directory", start_dir,
        )
        if path:
            self.temp_dir_edit.setText(path)

    def _auto_output_name(self, input_path: str):
        """Generate a default output filename based on input."""
        p = Path(input_path)
        fmt = self.format_combo.currentIndex()
        if fmt == 1:  # DeoVR Alpha
            suffix = f"_ALPHA{p.suffix}"
        else:
            suffix = f"_matte{p.suffix}"
        output = p.parent / f"{p.stem}{suffix}"
        self.output_edit.setText(str(output))

    def _show_video_info(self, path: str):
        """Display video metadata and auto-detect SBS."""
        try:
            info = get_video_info(path)
            self.info_label.setText(
                f"{info['width']}×{info['height']} | "
                f"{info['fps']} fps | "
                f"{info['num_frames']} frames | "
                f"{info['duration']}s | "
                f"{info['codec']}"
            )
            # Auto-detect SBS
            from vrautomatte.utils.sbs import detect_sbs
            is_sbs = detect_sbs(
                info["width"], info["height"]
            )
            self.sbs_check.setChecked(is_sbs)
            if is_sbs:
                self.sbs_auto_label.setText(
                    "(auto-detected)"
                )
            else:
                self.sbs_auto_label.setText("")

            # Enable scrubber for seek
            self.preview.set_scrubber_enabled(
                True, info["num_frames"]
            )
            self._video_info = info
        except Exception:
            self.info_label.setText(
                "Could not read video info"
            )

    def _on_format_changed(self, index: int):
        self.vr_row_widget.setVisible(index == 1)
        if self.input_edit.text():
            self._auto_output_name(self.input_edit.text())

    def _update_range_info(self):
        """Show duration hint based on start/end frame inputs."""
        if not self._video_info:
            self.range_info.setText("")
            return
        fps = float(self._video_info.get("fps", 30))
        total = self._video_info.get("num_frames", 0)
        start_text = self.start_frame_edit.text().strip()
        end_text = self.end_frame_edit.text().strip()
        if not start_text and not end_text:
            self.range_info.setText("")
            return
        start = int(start_text) if start_text else 1
        end = int(end_text) if end_text else total
        start = max(1, min(start, total))
        end = max(start, min(end, total))
        count = end - start + 1
        secs = count / fps if fps > 0 else 0
        self.range_info.setText(
            f"({count:,} frames, ~{secs:.1f}s)"
        )

    def _on_frame_scrubbed(self, frame_num: int):
        """Handle scrubber seek — extract and show source frame."""
        input_path = self.input_edit.text()
        if not input_path or not self._video_info:
            return

        try:
            import tempfile
            from vrautomatte.utils.ffmpeg import extract_frame

            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as tmp:
                tmp_path = tmp.name

            extract_frame(input_path, frame_num - 1, tmp_path)
            frame_img = Image.open(tmp_path).convert("RGB")
            frame_arr = np.array(frame_img)
            self.preview.update_preview(
                source_frame=frame_arr,
                frame_num=frame_num,
                total_frames=self._video_info["num_frames"],
            )

            import os
            os.unlink(tmp_path)
        except Exception as e:
            logger.debug(f"Scrubber seek failed: {e}")

    def _update_crf_label(self, value: int):
        self.crf_label.setText(str(value))

    def _update_fov_label(self, value: int):
        self.fov_label.setText(f"{value}°")

    def _toggle_theme(self):
        """Switch between light and dark themes."""
        self._is_dark = not self._is_dark
        self._colors = (
            DARK_COLORS if self._is_dark else LIGHT_COLORS
        )
        self._apply_theme()
        self._update_pov_warning()
        self._update_device_label()
        self._settings["dark_theme"] = self._is_dark

    def _apply_theme(self):
        """Apply the current theme stylesheet and colors."""
        style = DARK_STYLE if self._is_dark else LIGHT_STYLE
        QApplication.instance().setStyleSheet(style)
        self.theme_btn.setText(
            "☀️" if self._is_dark else "🌙"
        )
        c = self._colors
        mono = (
            "font-family: 'Cascadia Code', "
            "'Consolas', monospace;"
        )
        self.info_label.setStyleSheet(
            f"color: {c['info_label']}; font-size: 11px; "
            f"{mono}"
        )
        self.sbs_auto_label.setStyleSheet(
            f"color: {c['sbs_auto']}; font-size: 10px; "
            f"font-style: italic;"
        )
        self.pov_warning.setStyleSheet(
            f"color: {c['pov_warning_default']}; "
            f"font-size: 10px; font-style: italic;"
        )
        self.preview.apply_colors(c)
        self.range_info.setStyleSheet(
            f"color: {c['sbs_auto']}; font-size: 10px; "
            f"font-style: italic;"
        )

    def _on_model_changed(self, index: int):
        """Handle model combo selection change.

        If MatAnyone 2 is selected but not installed,
        trigger the install flow and revert to previous model.
        """
        if index == 2 and not self._ma2_available:
            # Revert to previous selection before install
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(0)
            self.model_combo.blockSignals(False)
            self._offer_install_matanyone2()
            return
        self._update_pov_warning()

    def _update_pov_warning(self, *_args):
        """Show info about POV quality per model."""
        c = self._colors
        if not self.pov_check.isChecked():
            self.pov_warning.setText("")
        elif self.model_combo.currentIndex() == 2:
            self.pov_warning.setText(
                "✓ Best quality (instance matting)"
            )
            self.pov_warning.setStyleSheet(
                f"color: {c['pov_quality']}; font-size: 10px; "
                f"font-style: italic;"
            )
        else:
            self.pov_warning.setText(
                "⚡ Fast mode (static mask subtraction)"
            )
            self.pov_warning.setStyleSheet(
                f"color: {c['pov_fast']}; font-size: 10px; "
                f"font-style: italic;"
            )

    def _update_device_label(self):
        """Update device info label, warn if CPU-only."""
        c = self._colors
        try:
            info = get_device_info()
            text = f"Device: {info['name']}"
            if "vram_gb" in info:
                text += f" ({info['vram_gb']} GB)"
            self.device_label.setText(text)

            if info["device"] == "cpu":
                self.device_label.setStyleSheet(
                    f"color: {c['device_cpu_warn']}; "
                    f"font-size: 11px;"
                )
                self.device_label.setText(
                    "⚠️ CPU mode — processing will be slower"
                )
        except Exception:
            self.device_label.setText("Device: unknown")

    def _build_config(
        self, input_path: str = "", output_path: str = ""
    ) -> PipelineConfig:
        """Build a PipelineConfig from the current UI state.

        Args:
            input_path: Override input (for batch mode).
            output_path: Override output (for batch mode).
        """
        ds_map = {0: 0.125, 1: 0.25, 2: 0.5, 3: 1.0}
        model_map = {
            0: "mobilenetv3",
            1: "resnet50",
            2: "matanyone2",
        }

        chunk_map = {0: 100, 1: 250, 2: 500, 3: 1000}

        config = PipelineConfig(
            input_path=input_path or self.input_edit.text(),
            output_path=output_path or self.output_edit.text(),
            model_variant=model_map.get(
                self.model_combo.currentIndex(), "mobilenetv3"
            ),
            downsample_ratio=ds_map.get(
                self.downsample_combo.currentIndex(), 0.25
            ),
            crf=self.crf_slider.value(),
            is_sbs=self.sbs_check.isChecked(),
            pov_mode=self.pov_check.isChecked(),
            temp_dir=self.temp_dir_edit.text(),
            chunk_size=chunk_map.get(
                self.chunk_size_combo.currentIndex(), 500
            ),
            auto_resume=self.resume_check.isChecked(),
        )

        # Frame range
        start_text = self.start_frame_edit.text().strip()
        end_text = self.end_frame_edit.text().strip()
        if start_text:
            try:
                config.start_frame = max(1, int(start_text))
            except ValueError:
                pass
        if end_text:
            try:
                config.end_frame = max(1, int(end_text))
            except ValueError:
                pass

        if self.format_combo.currentIndex() == 1:
            config.output_format = OutputFormat.DEOVR_ALPHA
            config.codec = (
                "libx265"
                if self.codec_combo.currentIndex() == 0
                else "libx264"
            )
            config.projection = (
                ProjectionType.EQUIRECTANGULAR
                if self.projection_combo.currentIndex() == 0
                else ProjectionType.FISHEYE
            )
            config.fisheye_fov = self.fov_slider.value()

            # Auto-resolve mask path
            mask = get_mask_path()
            if mask:
                config.fisheye_mask_path = str(mask)
        else:
            config.output_format = OutputFormat.MATTE_ONLY

        return config

    def _start_processing(self):
        """Validate inputs and start processing."""
        # Decide: batch mode or single file
        if self._batch_queue:
            self._start_batch()
            return

        if not self.input_edit.text():
            QMessageBox.warning(
                self, "Missing Input",
                "Please select an input video file.",
            )
            return
        if not self.output_edit.text():
            QMessageBox.warning(
                self, "Missing Output",
                "Please specify an output file path.",
            )
            return
        if not check_ffmpeg():
            self._show_ffmpeg_error()
            return

        # Check sam2 availability for features that need it
        needs_sam2 = (
            self.model_combo.currentIndex() == 2
            or self.pov_check.isChecked()
        )
        if needs_sam2 and not self._ma2_available:
            self._offer_install_matanyone2()
            return

        config = self._build_config()
        if not self._check_deovr_mask(config):
            return

        self._save_current_settings()
        self._run_single(config)

    def _start_batch(self):
        """Start processing the batch queue sequentially."""
        if not check_ffmpeg():
            self._show_ffmpeg_error()
            return

        self._batch_index = 0
        self._save_current_settings()
        self._run_next_batch_item()

    def _run_next_batch_item(self):
        """Pick the next item from the queue and process it."""
        if self._batch_index >= len(self._batch_queue):
            # All done
            self._unlock_ui()
            self.progress_bar.setValue(100)
            n = len(self._batch_queue)
            self.status_label.setText(
                f"Batch complete: {n} file{'s' if n != 1 else ''}"
            )
            QMessageBox.information(
                self, "Batch Complete",
                f"Successfully processed {n} files.",
            )
            return

        entry = self._batch_queue[self._batch_index]
        total = len(self._batch_queue)
        self.status_label.setText(
            f"Batch {self._batch_index + 1}/{total}: "
            f"{Path(entry['input']).name}"
        )
        # Highlight current item in list
        self.batch_list.setCurrentRow(self._batch_index)

        config = self._build_config(
            entry["input"], entry["output"]
        )
        if not self._check_deovr_mask(config):
            return
        self._run_single(config, is_batch=True)

    def _run_single(self, config: PipelineConfig,
                    is_batch: bool = False):
        """Launch the pipeline worker for a single file."""
        self._lock_ui()
        self.preview.clear()
        self.progress_bar.setValue(0)
        self.disk_label.setText("")

        self.worker = PipelineWorker(config)
        self.worker.progress.connect(self._on_progress)
        if is_batch:
            self.worker.finished.connect(
                self._on_batch_item_finished
            )
        else:
            self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _check_deovr_mask(self, config: PipelineConfig) -> bool:
        """Ensure the DeoVR mask is available, auto-downloading if needed.

        Returns:
            True if mask is available or not needed.
        """
        if config.output_format != OutputFormat.DEOVR_ALPHA:
            return True
        if config.projection != ProjectionType.EQUIRECTANGULAR:
            return True
        if config.fisheye_mask_path:
            return True

        # Try auto-download
        self.status_label.setText("Downloading DeoVR fisheye mask...")
        QApplication.processEvents()
        try:
            mask_path = ensure_mask()
            config.fisheye_mask_path = str(mask_path)
            self._settings["fisheye_mask_path"] = str(mask_path)
            return True
        except RuntimeError as e:
            QMessageBox.critical(
                self, "Mask Download Failed", str(e)
            )
            return False

    def _show_ffmpeg_error(self):
        QMessageBox.critical(
            self, "FFmpeg Not Found",
            "FFmpeg is required but was not found on your PATH.\n\n"
            "Please install FFmpeg:\n"
            "  Windows: winget install ffmpeg\n"
            "  Mac: brew install ffmpeg\n"
            "  Linux: sudo apt install ffmpeg",
        )

    def _offer_install_matanyone2(self):
        """Ask user to install MatAnyone 2 deps, then install."""
        feature = (
            "MatAnyone 2"
            if self.model_combo.currentIndex() == 2
            else "POV mode"
        )
        reply = QMessageBox.question(
            self,
            "Install MatAnyone 2?",
            f"{feature} requires MatAnyone 2 and SAM2 "
            f"which are not yet installed.\n\n"
            f"Download and install now?\n"
            f"(This may take a few minutes.)",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._run_install_matanyone2()

    def _run_install_matanyone2(self):
        """Run the MatAnyone 2 install in a background thread."""
        # Build a progress dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Installing MatAnyone 2…")
        dlg.setMinimumSize(560, 320)
        dlg.setModal(True)
        layout = QVBoxLayout(dlg)

        status = QLabel("Installing dependencies…")
        status.setStyleSheet(
            "font-weight: 600; font-size: 13px;"
        )
        layout.addWidget(status)

        log_box = QPlainTextEdit()
        log_box.setReadOnly(True)
        log_box.setStyleSheet(
            "font-family: 'Cascadia Code', 'Consolas', "
            "monospace; font-size: 11px;"
        )
        layout.addWidget(log_box)

        close_btn = QPushButton("Close")
        close_btn.setEnabled(False)
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        worker = InstallWorker()

        def on_output(line: str):
            log_box.appendPlainText(line)
            # Auto-scroll to bottom
            sb = log_box.verticalScrollBar()
            sb.setValue(sb.maximum())

        def on_finished(success: bool):
            close_btn.setEnabled(True)
            if success:
                status.setText("✅ Installation complete!")
                self._ma2_available = True
                self.model_combo.setItemText(
                    2, "MatAnyone 2 — single subject"
                )
                self.model_combo.setCurrentIndex(2)
            else:
                status.setText("❌ Installation failed")

        worker.output.connect(on_output)
        worker.finished.connect(on_finished)
        # Store ref to prevent GC
        dlg._worker = worker
        worker.start()
        dlg.exec()

    def _cancel_processing(self):
        if self.worker:
            self.worker.cancel()
            self.status_label.setText("Cancelling...")

    def _on_progress(self, p: PipelineProgress):
        """Handle progress updates from the worker thread."""
        if p.total_frames > 0:
            pct = int(
                ((p.stage_num - 1) / p.total_stages * 100)
                + (p.frame_num / p.total_frames
                   / p.total_stages * 100)
            )
            self.progress_bar.setValue(min(pct, 100))
        elif p.total_stages > 0:
            pct = int(p.stage_num / p.total_stages * 100)
            self.progress_bar.setValue(min(pct, 100))

        status = p.stage
        if p.total_frames > 0:
            status += (
                f" — frame {p.frame_num}/{p.total_frames}"
            )
        self.status_label.setText(status)

        if p.estimated_disk_gb > 0 and p.total_frames > 0:
            frac = p.frame_num / p.total_frames
            current = p.estimated_disk_gb * frac
            self.disk_label.setText(
                f"Disk: ~{current:.1f} / "
                f"~{p.estimated_disk_gb:.1f} GB estimated"
            )

        self.preview.update_preview(
            source_frame=p.source_frame,
            matte_frame=p.matte_frame,
            frame_num=p.frame_num,
            total_frames=p.total_frames,
            eta_sec=p.eta_sec,
            fps=p.fps,
            elapsed_sec=p.elapsed_sec,
        )

    def _on_finished(self, output_path: str):
        """Handle single-file completion."""
        self._unlock_ui()
        self.progress_bar.setValue(100)
        self.disk_label.setText("")
        self.status_label.setText(f"Complete: {output_path}")
        QMessageBox.information(
            self, "Processing Complete",
            f"Output saved to:\n{output_path}",
        )

    def _on_batch_item_finished(self, output_path: str):
        """Handle completion of one batch item, then start next."""
        logger.info(
            f"Batch item {self._batch_index + 1} done: {output_path}"
        )
        self._batch_index += 1
        self._run_next_batch_item()

    def _on_error(self, message: str):
        """Handle pipeline error."""
        self._unlock_ui()
        self.disk_label.setText("")
        self.status_label.setText(f"Error: {message}")
        if message != "Pipeline cancelled.":
            QMessageBox.critical(self, "Error", message)

    def _lock_ui(self):
        """Disable UI during processing."""
        self.start_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.input_btn.setEnabled(False)
        self.output_btn.setEnabled(False)
        self.add_batch_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.format_combo.setEnabled(False)
        self.crf_slider.setEnabled(False)
        self.downsample_combo.setEnabled(False)
        self.chunk_size_combo.setEnabled(False)
        self.resume_check.setEnabled(False)

    def _unlock_ui(self):
        """Re-enable UI after processing ends."""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.input_btn.setEnabled(True)
        self.output_btn.setEnabled(True)
        self.add_batch_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.format_combo.setEnabled(True)
        self.crf_slider.setEnabled(True)
        self.downsample_combo.setEnabled(True)
        self.chunk_size_combo.setEnabled(True)
        self.resume_check.setEnabled(True)
