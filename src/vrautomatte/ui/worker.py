"""Worker thread for running the pipeline without blocking the GUI."""

from PySide6.QtCore import QThread, Signal

from vrautomatte.pipeline.runner import Pipeline, PipelineConfig, PipelineProgress


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
