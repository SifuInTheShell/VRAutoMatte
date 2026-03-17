"""Side-by-side preview widget with frame scrubber."""

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def _numpy_to_pixmap(
    arr: np.ndarray, max_width: int = 400
) -> QPixmap:
    """Convert a numpy array to a QPixmap for display."""
    if arr is None:
        return QPixmap()

    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.ascontiguousarray(arr)

    h, w, c = arr.shape
    bytes_per_line = w * c
    qimg = QImage(
        arr.data, w, h, bytes_per_line,
        QImage.Format.Format_RGB888,
    )
    pixmap = QPixmap.fromImage(qimg)

    if w > max_width:
        pixmap = pixmap.scaledToWidth(
            max_width,
            Qt.TransformationMode.SmoothTransformation,
        )
    return pixmap


_MONO = (
    "font-family: 'Cascadia Code', 'Consolas', monospace;"
)


class PreviewWidget(QWidget):
    """Dual-pane preview: source frame and matte side by side.

    Signals:
        frame_scrubbed: Emitted when user drags the scrubber.
    """

    frame_scrubbed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._total_frames = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header row
        header_row = QHBoxLayout()
        self._header = QLabel("Preview")
        header_row.addWidget(self._header)
        header_row.addStretch()
        self.perf_label = QLabel("")
        header_row.addWidget(self.perf_label)
        layout.addLayout(header_row)

        # Image panes
        pane_layout = QHBoxLayout()
        pane_layout.setSpacing(8)

        source_pane = QVBoxLayout()
        self._source_header = QLabel("Source")
        self._source_header.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        source_pane.addWidget(self._source_header)
        self.source_label = QLabel()
        self.source_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.source_label.setMinimumSize(320, 180)
        source_pane.addWidget(self.source_label)
        pane_layout.addLayout(source_pane)

        matte_pane = QVBoxLayout()
        self._matte_header = QLabel("Matte")
        self._matte_header.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        matte_pane.addWidget(self._matte_header)
        self.matte_label = QLabel()
        self.matte_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.matte_label.setMinimumSize(320, 180)
        matte_pane.addWidget(self.matte_label)
        pane_layout.addLayout(matte_pane)

        layout.addLayout(pane_layout, stretch=1)

        # Scrubber row
        scrubber_row = QHBoxLayout()
        scrubber_row.setSpacing(8)

        self.frame_label = QLabel("—")
        self.frame_label.setFixedWidth(130)
        scrubber_row.addWidget(self.frame_label)

        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setRange(0, 0)
        self.scrubber.setEnabled(False)
        self.scrubber.valueChanged.connect(self._on_scrub)
        scrubber_row.addWidget(self.scrubber, stretch=1)

        self.eta_label = QLabel("")
        self.eta_label.setFixedWidth(150)
        self.eta_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
        )
        scrubber_row.addWidget(self.eta_label)

        layout.addLayout(scrubber_row)

    def apply_colors(self, c: dict) -> None:
        """Apply theme colors to all inline-styled widgets.

        Args:
            c: Color map from themes.py (LIGHT_COLORS or
               DARK_COLORS).
        """
        self._header.setStyleSheet(
            f"font-weight: 700; font-size: 12px; "
            f"letter-spacing: 0.5px; "
            f"text-transform: uppercase; "
            f"color: {c['preview_header']};"
        )
        self.perf_label.setStyleSheet(
            f"color: {c['preview_perf']}; font-size: 12px; "
            f"{_MONO} font-weight: 600;"
        )
        pane_hdr = (
            f"font-size: 10px; font-weight: 700; "
            f"letter-spacing: 1px; text-transform: uppercase; "
            f"color: {c['preview_pane_header']};"
        )
        self._source_header.setStyleSheet(pane_hdr)
        self._matte_header.setStyleSheet(pane_hdr)

        pane_bg = (
            f"background-color: {c['preview_pane_bg']}; "
            f"border: 1px solid {c['preview_pane_border']}; "
            f"border-radius: 8px;"
        )
        self.source_label.setStyleSheet(pane_bg)
        self.matte_label.setStyleSheet(pane_bg)

        mono_lbl = (
            f"color: {c['preview_mono']}; "
            f"font-size: 11px; {_MONO}"
        )
        self.frame_label.setStyleSheet(mono_lbl)
        self.eta_label.setStyleSheet(mono_lbl)

    def _on_scrub(self, value: int):
        self.frame_scrubbed.emit(value)

    def update_preview(
        self,
        source_frame: np.ndarray | None = None,
        matte_frame: np.ndarray | None = None,
        frame_num: int = 0,
        total_frames: int = 0,
        eta_sec: float = 0.0,
        fps: float = 0.0,
        elapsed_sec: float = 0.0,
    ) -> None:
        """Update the preview with new frame data."""
        if source_frame is not None:
            self.source_label.setPixmap(
                _numpy_to_pixmap(source_frame)
            )
        if matte_frame is not None:
            self.matte_label.setPixmap(
                _numpy_to_pixmap(matte_frame)
            )
        if total_frames > 0:
            self._total_frames = total_frames
            self.frame_label.setText(
                f"Frame {frame_num:,} / {total_frames:,}"
            )
            self.scrubber.blockSignals(True)
            self.scrubber.setRange(1, total_frames)
            self.scrubber.setValue(frame_num)
            self.scrubber.blockSignals(False)

        if eta_sec > 0:
            self.eta_label.setText(
                f"ETA: {_format_time(eta_sec)}"
            )
        elif elapsed_sec > 0:
            self.eta_label.setText(
                f"Elapsed: {_format_time(elapsed_sec)}"
            )
        if fps > 0:
            self.perf_label.setText(f"{fps:.1f} fps")

    def set_scrubber_enabled(
        self, enabled: bool, total_frames: int = 0
    ) -> None:
        """Enable/disable scrubber for manual seeking."""
        self.scrubber.setEnabled(enabled)
        if total_frames > 0:
            self.scrubber.setRange(1, total_frames)

    def clear(self) -> None:
        """Reset the preview to empty state."""
        self.source_label.clear()
        self.matte_label.clear()
        self.frame_label.setText("—")
        self.eta_label.setText("")
        self.perf_label.setText("")
        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, 0)
        self.scrubber.setValue(0)
        self.scrubber.setEnabled(False)
        self.scrubber.blockSignals(False)


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 0:
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins:02d}m"
