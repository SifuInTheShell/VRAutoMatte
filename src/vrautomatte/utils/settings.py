"""Settings persistence — save and restore user preferences."""

import json
from pathlib import Path

from loguru import logger

_SETTINGS_DIR = Path.home() / ".config" / "vrautomatte"
_SETTINGS_FILE = _SETTINGS_DIR / "settings.json"

_DEFAULTS = {
    "model_variant": 0,          # 0=mobilenetv3, 1=resnet50, 2=matanyone2
    "downsample_ratio": 1,       # combo index
    "crf": 18,
    "output_format": 0,          # 0=matte only, 1=deovr alpha
    "projection": 0,             # 0=equirect→fisheye, 1=already fisheye
    "fisheye_fov": 180,
    "codec": 0,                  # 0=HEVC, 1=H.264
    "last_input_dir": "",
    "last_output_dir": "",
    "window_width": 900,
    "window_height": 780,
    "fisheye_mask_path": "",
}


def load_settings() -> dict:
    """Load settings from disk, falling back to defaults.

    Returns:
        Settings dictionary.
    """
    settings = dict(_DEFAULTS)
    if _SETTINGS_FILE.exists():
        try:
            with open(_SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            settings.update(saved)
            logger.debug(f"Loaded settings from {_SETTINGS_FILE}")
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")
    return settings


def save_settings(settings: dict) -> None:
    """Save settings to disk.

    Args:
        settings: Settings dictionary to persist.
    """
    try:
        _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        logger.debug(f"Saved settings to {_SETTINGS_FILE}")
    except Exception as e:
        logger.warning(f"Failed to save settings: {e}")
