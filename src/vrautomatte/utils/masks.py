"""Auto-download and management of the DeoVR fisheye mask."""

import urllib.request
from pathlib import Path

from loguru import logger

# DeoVR provides mask files at various resolutions.
# The mask8k.png is used for the equirect → fisheye overlay.
_MASK_URL = (
    "http://rest.s3for.me/insights.deovr.com/images/mask8k.png"
)
_FALLBACK_URLS = [
    "https://deovr.com/static/masks/mask8k.png",
]


def _mask_dir() -> Path:
    """Return the directory where masks are cached."""
    d = Path.home() / ".cache" / "vrautomatte" / "masks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_mask_path() -> Path | None:
    """Get the path to the cached DeoVR fisheye mask.

    Returns:
        Path to the mask file if it exists, None otherwise.
    """
    mask = _mask_dir() / "mask8k.png"
    return mask if mask.exists() else None


def download_mask(on_progress=None) -> Path:
    """Download the DeoVR fisheye mask (mask8k.png).

    Args:
        on_progress: Optional callback(bytes_read, total_bytes).

    Returns:
        Path to the downloaded mask file.

    Raises:
        RuntimeError: If download fails from all sources.
    """
    dest = _mask_dir() / "mask8k.png"
    if dest.exists():
        logger.debug(f"Mask already cached: {dest}")
        return dest

    urls = [_MASK_URL] + _FALLBACK_URLS
    last_error = None

    for url in urls:
        try:
            logger.info(f"Downloading DeoVR mask from {url}...")

            def _reporthook(count, block_size, total_size):
                if on_progress and total_size > 0:
                    on_progress(count * block_size, total_size)

            urllib.request.urlretrieve(url, str(dest), reporthook=_reporthook)

            # Verify the file is a valid PNG (check magic bytes)
            with open(dest, "rb") as f:
                magic = f.read(8)
            if magic[:4] != b"\x89PNG":
                dest.unlink(missing_ok=True)
                raise ValueError("Downloaded file is not a valid PNG")

            logger.info(f"Mask saved to {dest}")
            return dest
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to download from {url}: {e}")
            dest.unlink(missing_ok=True)

    raise RuntimeError(
        f"Failed to download DeoVR mask. Last error: {last_error}\n\n"
        f"You can manually download mask8k.png from:\n"
        f"https://deovr.com/blog/123-converting-equirectangular-vr-footage-into-fisheye\n"
        f"and place it in: {_mask_dir()}"
    )


def ensure_mask() -> Path:
    """Ensure the mask is available, downloading if necessary.

    Returns:
        Path to the mask file.
    """
    existing = get_mask_path()
    if existing:
        return existing
    return download_mask()
