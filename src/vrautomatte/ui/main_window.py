"""Main application window for VRAutoMatte."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from vrautomatte.pipeline.runner import (
    OutputFormat,
    PipelineConfig,
    PipelineProgress,
    ProjectionType,
)
from vrautomatte.ui.preview import PreviewWidget
from vrautomatte.ui.worker import PipelineWorker
from vrautomatte.utils.ffmpeg import check_ffmpeg, get_video_info
from vrautomatte.utils.gpu import get_device_info


DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0f0f1a;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2a2a3e;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #b0b0cc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QLineEdit {
    background-color: #1a1a2e;
    border: 1px solid #333355;
    border-radius: 4px;
    padding: 6px 10px;
    color: #e0e0e0;
    selection-background-color: #4a4a8e;
}
QLineEdit:focus {
    border-color: #6666aa;
}
QPushButton {
    background-color: #2a2a4e;
    border: 1px solid #3a3a5e;
    border-radius: 4px;
    padding: 6px 16px;
    color: #d0d0ee;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #3a3a6e;
    border-color: #5a5a8e;
}
QPushButton:pressed {
    background-color: #1a1a3e;
}
QPushButton:disabled {
    background-color: #1a1a2e;
    color: #555;
    border-color: #222;
}
QPushButton#startButton {
    background-color: #2e5a2e;
    border-color: #3e7a3e;
    color: #c0eec0;
    font-size: 14px;
    padding: 8px 24px;
}
QPushButton#startButton:hover {
    background-color: #3e7a3e;
}
QPushButton#cancelButton {
    background-color: #5a2e2e;
    border-color: #7a3e3e;
    color: #eec0c0;
}
QPushButton#cancelButton:hover {
    background-color: #7a3e3e;
}
QComboBox {
    background-color: #1a1a2e;
    border: 1px solid #333355;
    border-radius: 4px;
    padding: 4px 8px;
    color: #e0e0e0;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #1a1a2e;
    border: 1px solid #333355;
    color: #e0e0e0;
    selection-background-color: #3a3a6e;
}
QSlider::groove:horizontal {
    border: 1px solid #333;
    height: 6px;
    background: #1a1a2e;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #6666aa;
    border: 1px solid #8888cc;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #4a4a8e;
    border-radius: 3px;
}
QProgressBar {
    border: 1px solid #333355;
    border-radius: 4px;
    text-align: center;
    background-color: #1a1a2e;
    color: #b0b0cc;
    height: 22px;
}
QProgressBar::chunk {
    background-color: #4a4a8e;
    border-radius: 3px;
}
QLabel#statusLabel {
    color: #888;
    font-size: 12px;
}
QLabel#deviceLabel {
    color: #6a6a9e;
    font-size: 11px;
}
"""


class MainWindow(QMainWindow):
    """Main application window.

    Layout:
    - File I/O section (input file, output file)
    - Settings section (model, quality, projection, FOV)
    - Preview section (source frame | generated matte)
    - Action bar (Start button, progress bar, status)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VRAutoMatte")
        self.setMinimumSize(800, 700)
        self.resize(900, 780)
        self.worker: PipelineWorker | None = None
        self._setup_ui()
        self._update_device_label()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(16, 16, 16, 16)

        # ── File I/O ──
        io_group = QGroupBox("Files")
        io_layout = QVBoxLayout(io_group)

        # Input row
        in_row = QHBoxLayout()
        in_row.addWidget(QLabel("Input:"))
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select input video file...")
        self.input_edit.setReadOnly(True)
        in_row.addWidget(self.input_edit, stretch=1)
        self.input_btn = QPushButton("Browse...")
        self.input_btn.clicked.connect(self._browse_input)
        in_row.addWidget(self.input_btn)
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
        self.info_label.setStyleSheet("color: #6a6a9e; font-size: 11px;")
        io_layout.addWidget(self.info_label)

        root.addWidget(io_group)

        # ── Settings ──
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Row 1: Model + Output Format
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Matting Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["mobilenetv3 (fast)", "resnet50 (quality)"])
        row1.addWidget(self.model_combo)
        row1.addSpacing(20)
        row1.addWidget(QLabel("Output Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Matte Only", "DeoVR Alpha Pack"])
        self.format_combo.currentIndexChanged.connect(
            self._on_format_changed
        )
        row1.addWidget(self.format_combo)
        row1.addStretch()
        settings_layout.addLayout(row1)

        # Row 2: Quality (CRF) slider
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Quality (CRF):"))
        self.crf_slider = QSlider(Qt.Orientation.Horizontal)
        self.crf_slider.setRange(10, 30)
        self.crf_slider.setValue(18)
        self.crf_slider.setTickInterval(2)
        self.crf_slider.setTickPosition(
            QSlider.TickPosition.TicksBelow
        )
        self.crf_slider.valueChanged.connect(self._update_crf_label)
        row2.addWidget(self.crf_slider)
        self.crf_label = QLabel("18")
        self.crf_label.setFixedWidth(30)
        row2.addWidget(self.crf_label)
        row2.addSpacing(20)

        row2.addWidget(QLabel("Downsample:"))
        self.downsample_combo = QComboBox()
        self.downsample_combo.addItems(
            ["0.125 (fastest)", "0.25 (balanced)", "0.5 (quality)",
             "1.0 (full res)"]
        )
        self.downsample_combo.setCurrentIndex(1)
        row2.addWidget(self.downsample_combo)
        row2.addStretch()
        settings_layout.addLayout(row2)

        # Row 3: VR-specific (conditional)
        self.vr_row_widget = QWidget()
        vr_row = QHBoxLayout(self.vr_row_widget)
        vr_row.setContentsMargins(0, 0, 0, 0)
        vr_row.addWidget(QLabel("Projection:"))
        self.projection_combo = QComboBox()
        self.projection_combo.addItems(
            ["Equirectangular → Fisheye", "Already Fisheye"]
        )
        vr_row.addWidget(self.projection_combo)
        vr_row.addSpacing(20)
        vr_row.addWidget(QLabel("Fisheye FOV:"))
        self.fov_slider = QSlider(Qt.Orientation.Horizontal)
        self.fov_slider.setRange(170, 210)
        self.fov_slider.setValue(180)
        self.fov_slider.valueChanged.connect(self._update_fov_label)
        vr_row.addWidget(self.fov_slider)
        self.fov_label = QLabel("180°")
        self.fov_label.setFixedWidth(35)
        vr_row.addWidget(self.fov_label)
        vr_row.addSpacing(20)
        vr_row.addWidget(QLabel("Codec:"))
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(["HEVC (H.265)", "H.264"])
        vr_row.addWidget(self.codec_combo)
        vr_row.addStretch()
        self.vr_row_widget.setVisible(False)  # hidden until DeoVR selected
        settings_layout.addWidget(self.vr_row_widget)

        root.addWidget(settings_group)

        # ── Preview ──
        self.preview = PreviewWidget()
        root.addWidget(self.preview, stretch=1)

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

        # Status bar
        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        self.device_label = QLabel("")
        self.device_label.setObjectName("deviceLabel")
        status_row.addWidget(self.device_label)
        root.addLayout(status_row)

    # ── Slots ──

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Video",
            "", "Video Files (*.mp4 *.mkv *.mov *.avi *.webm);;All Files (*)",
        )
        if path:
            self.input_edit.setText(path)
            self._auto_output_name(path)
            self._show_video_info(path)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output As",
            self.output_edit.text() or "",
            "Video Files (*.mp4 *.mkv);;All Files (*)",
        )
        if path:
            self.output_edit.setText(path)

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
        """Display video metadata below the input field."""
        try:
            info = get_video_info(path)
            self.info_label.setText(
                f"{info['width']}×{info['height']} | "
                f"{info['fps']} fps | "
                f"{info['num_frames']} frames | "
                f"{info['duration']}s | "
                f"{info['codec']}"
            )
        except Exception:
            self.info_label.setText("Could not read video info")

    def _on_format_changed(self, index: int):
        self.vr_row_widget.setVisible(index == 1)
        # Re-generate output name
        if self.input_edit.text():
            self._auto_output_name(self.input_edit.text())

    def _update_crf_label(self, value: int):
        self.crf_label.setText(str(value))

    def _update_fov_label(self, value: int):
        self.fov_label.setText(f"{value}°")

    def _update_device_label(self):
        try:
            info = get_device_info()
            text = f"Device: {info['name']}"
            if "vram_gb" in info:
                text += f" ({info['vram_gb']} GB)"
            self.device_label.setText(text)
        except Exception:
            self.device_label.setText("Device: unknown")

    def _build_config(self) -> PipelineConfig:
        """Build a PipelineConfig from the current UI state."""
        ds_map = {0: 0.125, 1: 0.25, 2: 0.5, 3: 1.0}
        model_map = {0: "mobilenetv3", 1: "resnet50"}

        config = PipelineConfig(
            input_path=self.input_edit.text(),
            output_path=self.output_edit.text(),
            model_variant=model_map.get(
                self.model_combo.currentIndex(), "mobilenetv3"
            ),
            downsample_ratio=ds_map.get(
                self.downsample_combo.currentIndex(), 0.25
            ),
            crf=self.crf_slider.value(),
        )

        if self.format_combo.currentIndex() == 1:
            config.output_format = OutputFormat.DEOVR_ALPHA
            config.codec = (
                "libx265" if self.codec_combo.currentIndex() == 0
                else "libx264"
            )
            config.projection = (
                ProjectionType.EQUIRECTANGULAR
                if self.projection_combo.currentIndex() == 0
                else ProjectionType.FISHEYE
            )
            config.fisheye_fov = self.fov_slider.value()
        else:
            config.output_format = OutputFormat.MATTE_ONLY

        return config

    def _start_processing(self):
        """Validate inputs and start the pipeline worker."""
        if not self.input_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an input video file.")
            return

        if not self.output_edit.text():
            QMessageBox.warning(self, "Missing Output", "Please specify an output file path.")
            return

        if not check_ffmpeg():
            QMessageBox.critical(
                self, "FFmpeg Not Found",
                "FFmpeg is required but was not found on your PATH.\n\n"
                "Please install FFmpeg:\n"
                "  Windows: winget install ffmpeg\n"
                "  Mac: brew install ffmpeg\n"
                "  Linux: sudo apt install ffmpeg",
            )
            return

        config = self._build_config()

        # Check DeoVR-specific requirements
        if config.output_format == OutputFormat.DEOVR_ALPHA:
            if (config.projection == ProjectionType.EQUIRECTANGULAR
                    and not config.fisheye_mask_path):
                # TODO: auto-download mask or bundle it
                QMessageBox.warning(
                    self, "Missing Fisheye Mask",
                    "DeoVR alpha packing requires a fisheye mask file "
                    "(mask8k.png).\n\n"
                    "Download it from the DeoVR documentation and set "
                    "the path in settings.",
                )
                return

        # Lock UI
        self.start_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")
        self.preview.clear()

        # Start worker
        self.worker = PipelineWorker(config)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _cancel_processing(self):
        if self.worker:
            self.worker.cancel()
            self.status_label.setText("Cancelling...")

    def _on_progress(self, p: PipelineProgress):
        """Handle progress updates from the worker thread."""
        # Update progress bar
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

        # Update status
        status = p.stage
        if p.total_frames > 0:
            status += f" — frame {p.frame_num}/{p.total_frames}"
        self.status_label.setText(status)

        # Update preview
        self.preview.update_preview(
            source_frame=p.source_frame,
            matte_frame=p.matte_frame,
            frame_num=p.frame_num,
            total_frames=p.total_frames,
        )

    def _on_finished(self, output_path: str):
        """Handle pipeline completion."""
        self._unlock_ui()
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Complete: {output_path}")
        QMessageBox.information(
            self, "Processing Complete",
            f"Output saved to:\n{output_path}",
        )

    def _on_error(self, message: str):
        """Handle pipeline error."""
        self._unlock_ui()
        self.status_label.setText(f"Error: {message}")
        if message != "Pipeline cancelled.":
            QMessageBox.critical(self, "Error", message)

    def _unlock_ui(self):
        """Re-enable UI after processing ends."""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
