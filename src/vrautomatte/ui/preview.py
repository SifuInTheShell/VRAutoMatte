"""Side-by-side preview widget showing source frame and generated matte."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


def _numpy_to_pixmap(arr: np.ndarray, max_width: int = 400) -> QPixmap:
    """Convert a numpy array (RGB or grayscale) to a QPixmap.

    Args:
        arr: Image array — (H, W, 3) for RGB or (H, W) for grayscale.
        max_width: Scale down to this max width for display.

    Returns:
        QPixmap ready for display.
    """
    if arr.ndim == 2:
        # Grayscale → RGB for display
        arr = np.stack([arr, arr, arr], axis=-1)

    h, w, c = arr.shape
    bytes_per_line = w * c
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)

    if w > max_width:
        pixmap = pixmap.scaledToWidth(
            max_width, Qt.TransformationMode.SmoothTransformation
        )
    return pixmap


class PreviewWidget(QWidget):
    """Dual-pane preview showing source frame and matte side by side.

    The widget displays a source video frame on the left and the
    corresponding generated alpha matte on the right, along with a
    frame counter below.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Preview")
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)

        # Image panes
        pane_layout = QHBoxLayout()

        self.source_label = QLabel("Source")
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_label.setMinimumSize(320, 180)
        self.source_label.setStyleSheet(
            "background-color: #1a1a2e; border: 1px solid #333; "
            "border-radius: 4px; color: #666;"
        )

        self.matte_label = QLabel("Matte")
        self.matte_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.matte_label.setMinimumSize(320, 180)
        self.matte_label.setStyleSheet(
            "background-color: #1a1a2e; border: 1px solid #333; "
            "border-radius: 4px; color: #666;"
        )

        pane_layout.addWidget(self.source_label)
        pane_layout.addWidget(self.matte_label)
        layout.addLayout(pane_layout)

        # Frame counter
        self.frame_label = QLabel("")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self.frame_label)

    def update_preview(
        self,
        source_frame: np.ndarray | None,
        matte_frame: np.ndarray | None,
        frame_num: int = 0,
        total_frames: int = 0,
    ) -> None:
        """Update the preview with new frame data.

        Args:
            source_frame: RGB numpy array (H, W, 3) or None.
            matte_frame: Grayscale numpy array (H, W) or None.
            frame_num: Current frame number (1-based).
            total_frames: Total number of frames.
        """
        if source_frame is not None:
            pixmap = _numpy_to_pixmap(source_frame)
            self.source_label.setPixmap(pixmap)

        if matte_frame is not None:
            pixmap = _numpy_to_pixmap(matte_frame)
            self.matte_label.setPixmap(pixmap)

        if total_frames > 0:
            self.frame_label.setText(
                f"Frame {frame_num} / {total_frames}"
            )

    def clear(self) -> None:
        """Reset the preview to empty state."""
        self.source_label.clear()
        self.source_label.setText("Source")
        self.matte_label.clear()
        self.matte_label.setText("Matte")
        self.frame_label.setText("")
