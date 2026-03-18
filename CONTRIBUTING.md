# Contributing to VRAutoMatte

## Development Setup

```bash
git clone https://github.com/SifuInTheShell/VRAutoMatte.git
cd VRAutoMatte
uv sync --extra matanyone2   # or just `uv sync` for RVM-only
```

## Running Tests

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

All 50 tests should pass. Tests mock GPU and filesystem operations — no GPU or FFmpeg required to run them.

## Code Style

- **Logging**: `from loguru import logger` — no `print()`
- **Strings**: double quotes
- **Line length**: 100 characters max
- **Imports**: stdlib → third-party → local, alphabetical within groups
- **Type hints**: required for new/modified functions
- **Module size**: target ≤ 150 LOC, hard cap 200 LOC (CSS/theme files exempt)
- **Docstrings**: required for modules, classes, and public functions

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new matting model support
fix: correct SBS frame merge for odd-width videos
docs: update GPU requirements table
test: add scene change detector edge cases
style: adjust light theme contrast
```

## Adding a New Matting Model

1. Create `src/vrautomatte/pipeline/your_model.py`
2. Implement the `MatteProcessor` protocol (see `matte.py`)
3. Add the variant to `create_processor()` factory in `matte.py`
4. Add the model name to the UI combo box in `main_window.py`
5. Write tests in `tests/test_your_model.py`

## Project Structure

See the [Architecture section](README.md#architecture) in the README.
