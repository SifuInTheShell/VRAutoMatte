"""VRAutoMatte application entry point."""

import sys

from loguru import logger
from PySide6.QtWidgets import QApplication

from vrautomatte.ui.main_window import DARK_STYLE, MainWindow


def main():
    """Launch the VRAutoMatte GUI application."""
    # Configure loguru
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | "
                      "<level>{level: <8}</level> | "
                      "<cyan>{message}</cyan>")

    logger.info("Starting VRAutoMatte...")

    app = QApplication(sys.argv)
    app.setApplicationName("VRAutoMatte")
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
