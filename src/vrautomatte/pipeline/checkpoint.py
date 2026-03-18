"""Pipeline checkpoint for resumable processing.

Saves progress after each segment flush so that long jobs
can survive crashes or cancellation without restarting from
frame 1.

Validates against the input file (first-64KB hash) and the
pipeline config (serialized hash) to detect stale state.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger

_FILENAME = "checkpoint.json"


@dataclass
class PipelineCheckpoint:
    """Checkpoint state for a pipeline run.

    Attributes:
        input_path: Absolute path to the input video.
        input_hash: SHA-256 of the first 64 KB of the input.
        config_hash: SHA-256 of the serialized config dict.
        total_frames: Total frames to process.
        chunk_size: Frames per extraction chunk.
        completed_segments: Segments successfully flushed.
        completed_frames: Total frames processed so far.
        timestamp: ISO timestamp of the last save.
    """

    input_path: str
    input_hash: str
    config_hash: str
    total_frames: int
    chunk_size: int
    completed_segments: int
    completed_frames: int
    timestamp: str

    def save(self, tmp_dir: Path) -> None:
        """Write checkpoint to disk."""
        path = tmp_dir / _FILENAME
        data = asdict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(
            f"Checkpoint saved: seg={self.completed_segments}, "
            f"frames={self.completed_frames}"
        )

    @classmethod
    def load(cls, tmp_dir: Path) -> "PipelineCheckpoint | None":
        """Load checkpoint if one exists and is valid JSON.

        Returns None if no file or if malformed.
        """
        path = tmp_dir / _FILENAME
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls(**data)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Invalid checkpoint file: {e}")
            return None

    def validate(
        self, input_path: Path, config_hash: str,
    ) -> bool:
        """Check that checkpoint matches the current run.

        Returns False if input or config has changed.
        """
        current_hash = hash_file_head(input_path)
        if self.input_hash != current_hash:
            logger.info("Checkpoint stale: input file changed")
            return False
        if self.config_hash != config_hash:
            logger.info("Checkpoint stale: config changed")
            return False
        return True

    @classmethod
    def delete(cls, tmp_dir: Path) -> None:
        """Remove the checkpoint file."""
        (tmp_dir / _FILENAME).unlink(missing_ok=True)


def hash_file_head(path: Path, size: int = 65536) -> str:
    """SHA-256 of the first ``size`` bytes of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(size))
    return h.hexdigest()


def hash_config(config) -> str:
    """SHA-256 of PipelineConfig's processing-relevant fields.

    Excludes paths and temp settings that don't affect output.
    """
    d = asdict(config)
    for key in ("input_path", "output_path", "temp_dir"):
        d.pop(key, None)
    serialized = json.dumps(d, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def deterministic_temp_name(
    input_path: Path, config_hash: str,
) -> str:
    """Generate a deterministic temp directory name."""
    stem = input_path.stem
    safe_stem = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in stem
    )[:50]
    return f"vrautomatte_{safe_stem}_{config_hash[:8]}"


def cleanup_stale_dirs(
    base_dir: Path, max_age_days: int = 7,
) -> None:
    """Remove vrautomatte temp dirs older than max_age_days."""
    if not base_dir.exists():
        return
    import shutil

    cutoff = time.time() - max_age_days * 86400
    for d in base_dir.iterdir():
        if (
            d.is_dir()
            and d.name.startswith("vrautomatte_")
            and d.stat().st_mtime < cutoff
        ):
            try:
                shutil.rmtree(d)
                logger.info(
                    f"Cleaned stale temp dir: {d.name}"
                )
            except OSError as e:
                logger.debug(
                    f"Could not remove {d.name}: {e}"
                )
