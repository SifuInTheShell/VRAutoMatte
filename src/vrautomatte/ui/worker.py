"""Worker threads for pipeline execution and dependency installation."""

import subprocess
import sys

from PySide6.QtCore import QThread, Signal

from vrautomatte.pipeline.runner import (
    Pipeline,
    PipelineConfig,
    PipelineProgress,
)


class PipelineWorker(QThread):
    """Runs the matting pipeline in a background thread.

    Signals:
        progress: Emitted with PipelineProgress updates.
        finished: Emitted with the output file path on success.
        error: Emitted with the error message on failure.
    """

    progress = Signal(object)   # PipelineProgress
    finished = Signal(str)      # output path
    error = Signal(str)         # error message

    def __init__(self, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._pipeline: Pipeline | None = None

    def run(self):
        """Execute the pipeline (called by QThread.start())."""
        try:
            self._pipeline = Pipeline(
                config=self.config,
                on_progress=self._on_progress,
            )
            output_path = self._pipeline.run()
            self.finished.emit(str(output_path))
        except InterruptedError:
            self.error.emit("Pipeline cancelled.")
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        """Request pipeline cancellation."""
        if self._pipeline:
            self._pipeline.cancel()

    def _on_progress(self, p: PipelineProgress):
        """Forward progress from the pipeline to the GUI thread."""
        self.progress.emit(p)


class InstallWorker(QThread):
    """Install optional dependencies in a background thread.

    Runs `uv sync --extra matanyone2` (or pip fallback)
    and streams output line-by-line.

    Signals:
        output: Emitted per line of install output.
        finished: Emitted with True on success, False on failure.
    """

    output = Signal(str)
    finished = Signal(bool)

    def run(self):
        """Run the install command."""
        try:
            cmd = self._build_command()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                self.output.emit(line.rstrip())
            process.wait()
            self.finished.emit(process.returncode == 0)
        except Exception as e:
            self.output.emit(f"Install failed: {e}")
            self.finished.emit(False)

    _PACKAGES = [
        "matanyone2 @ git+https://github.com/pq-yang/MatAnyone2.git",
        "sam2>=1.0",
    ]

    @staticmethod
    def _build_command() -> list[str]:
        """Build the install command, preferring uv.

        Uses ``uv pip install`` instead of ``uv sync`` so that
        already-loaded DLLs (numpy, torch, etc.) are not replaced
        while the application is running.
        """
        try:
            subprocess.run(
                ["uv", "--version"],
                capture_output=True, check=True,
            )
            return [
                "uv", "pip", "install",
                *InstallWorker._PACKAGES,
            ]
        except (FileNotFoundError, subprocess.CalledProcessError):
            return [
                sys.executable, "-m", "pip", "install",
                *InstallWorker._PACKAGES,
            ]
