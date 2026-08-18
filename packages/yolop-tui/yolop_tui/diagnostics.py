from __future__ import annotations

import logging
from pathlib import Path

_LOGGER = logging.getLogger("yolop_tui")
_HANDLER_PATH = "_yolop_tui_handler_path"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(workspace: str | Path) -> Path:
    """Configure diagnostic logging below one trusted local workspace."""
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Workspace directory does not exist: {root}")

    path = root / ".yolop" / "yolop.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path = str(path)

    for handler in list(_LOGGER.handlers):
        if getattr(handler, _HANDLER_PATH, None) != resolved_path:
            continue
        return path

    for handler in list(_LOGGER.handlers):
        if not hasattr(handler, _HANDLER_PATH):
            continue
        _LOGGER.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(path, encoding="utf-8")
    setattr(handler, _HANDLER_PATH, resolved_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.addHandler(handler)
    _LOGGER.info("YoloP TUI logging started")
    return path


__all__ = ["configure_logging"]
