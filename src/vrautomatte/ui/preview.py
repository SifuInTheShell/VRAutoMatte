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


def _numpy_to_pixmap(arr: np.ndarray, max_width: int = 400) -> QPixmap:
    """Convert a numpy array (RGB or grayscale) to a QPixmap.

    Args:
        arr: Image array — (H, W, 3) for RGB or (H, W) for grayscale.
        max_width: Scale down to this max width for display.

    Returns:
        QPixmap ready for display.
    """
    if arr is None:
        return QPixmap()

    # Ensure contiguous memory layout
    arr = np.ascontiguousarray(arr)

    if arr.ndim == 2:
        # Grayscale → RGB for display
        arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.ascontiguousarray(arr)

    h, w, c = arr.shape
    bytes_per_line = w * c
    qimg = QImage(
        arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
    )
    pixmap = QPixmap.fromImage(qimg)

    if w > max_width:
        pixmap = pixmap.scaledToWidth(
            max_width, Qt.TransformationMode.SmoothTransformation
        )
    return pixmap


class PreviewWidget(QWidget):
    """Dual-pane preview showing source frame and matte side by side.

    Features:
    - Source frame on the left, generated matte on the right
    - Frame counter with scrubber slider
    - ETA and FPS display during processing

    Signals:
        frame_scrubbed: Emitted when user drags the scrubber (frame_num).
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
        header = QLabel("Preview")
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_row.addWidget(header)
        header_row.addStretch()
        self.perf_label = QLabel("")
        self.perf_label.setStyleSheet(
            "color: #6a9a6a; font-size: 11px; font-family: monospace;"
        )
        header_row.addWidget(self.perf_label)
        layout.addLayout(header_row)

        # Image panes
        pane_layout = QHBoxLayout()
        pane_layout.setSpacing(8)

        # Source pane
        source_pane = QVBoxLayout()
        source_header = QLabel("Source Frame")
        source_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        source_header.setStyleSheet(
            "color: #8888aa; font-size: 11px; font-weight: bold;"
        )
        source_pane.addWidget(source_header)
        self.source_label = QLabel()
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_label.setMinimumSize(320, 180)
        self.source_label.setStyleSheet(
            "background-color: #12121f; border: 1px solid #2a2a3e; "
            "border-radius: 4px;"
        )
        source_pane.addWidget(self.source_label)
        pane_layout.addLayout(source_pane)

        # Matte pane
        matte_pane = QVBoxLayout()
        matte_header = QLabel("Generated Matte")
        matte_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        matte_header.setStyleSheet(
            "color: #8888aa; font-size: 11px; font-weight: bold;"
        )
        matte_pane.addWidget(matte_header)
        self.matte_label = QLabel()
        self.matte_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.matte_label.setMinimumSize(320, 180)
        self.matte_label.setStyleSheet(
            "background-color: #12121f; border: 1px solid #2a2a3e; "
            "border-radius: 4px;"
        )
        matte_pane.addWidget(self.matte_label)
        pane_layout.addLayout(matte_pane)

        layout.addLayout(pane_layout, stretch=1)

        # Scrubber row
        scrubber_row = QHBoxLayout()
        scrubber_row.setSpacing(8)

        self.frame_label = QLabel("—")
        self.frame_label.setFixedWidth(120)
        self.frame_label.setStyleSheet(
            "color: #888; font-size: 12px; font-family: monospace;"
        )
        scrubber_row.addWidget(self.frame_label)

        self.scrubber = QSlider(Qt.Orientation.Horizontal)
        self.scrubber.setRange(0, 0)
        self.scrubber.setEnabled(False)
        self.scrubber.valueChanged.connect(self._on_scrub)
        scrubber_row.addWidget(self.scrubber, stretch=1)

        self.eta_label = QLabel("")
        self.eta_label.setFixedWidth(140)
        self.eta_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.eta_label.setStyleSheet(
            "color: #888; font-size: 12px; font-family: monospace;"
        )
        scrubber_row.addWidget(self.eta_label)

        layout.addLayout(scrubber_row)

    def _on_scrub(self, value: int):
        """Handle scrubber drag."""
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
        """Update the preview with new frame data.

        Args:
            source_frame: RGB numpy array (H, W, 3) or None.
            matte_frame: Grayscale numpy array (H, W) or None.
            frame_num: Current frame number (1-based).
            total_frames: Total number of frames.
            eta_sec: Estimated time remaining in seconds.
            fps: Current processing frames per second.
            elapsed_sec: Total elapsed time in seconds.
        """
        if source_frame is not None:
            pixmap = _numpy_to_pixmap(source_frame)
            self.source_label.setPixmap(pixmap)

        if matte_frame is not None:
            pixmap = _numpy_to_pixmap(matte_frame)
            self.matte_label.setPixmap(pixmap)

        if total_frames > 0:
            self._total_frames = total_frames
            self.frame_label.setText(
                f"Frame {frame_num:,} / {total_frames:,}"
            )
            # Update scrubber position without triggering signal
            self.scrubber.blockSignals(True)
            self.scrubber.setRange(1, total_frames)
            self.scrubber.setValue(frame_num)
            self.scrubber.blockSignals(False)

        # ETA display
        if eta_sec > 0:
            eta_str = _format_time(eta_sec)
            self.eta_label.setText(f"ETA: {eta_str}")
        elif elapsed_sec > 0:
            self.eta_label.setText(
                f"Elapsed: {_format_time(elapsed_sec)}"
            )

        # FPS display
        if fps > 0:
            self.perf_label.setText(f"{fps:.1f} fps")

    def set_scrubber_enabled(self, enabled: bool,
                             total_frames: int = 0) -> None:
        """Enable or disable the scrubber for manual seeking.

        Args:
            enabled: Whether scrubber is interactive.
            total_frames: Total frame count for the range.
        """
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
    """Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string like '2m 34s' or '1h 12m'.
    """
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
