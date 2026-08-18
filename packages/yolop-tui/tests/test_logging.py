from __future__ import annotations

import logging
from pathlib import Path

from yolop_tui.diagnostics import configure_logging


def test_configure_logging_writes_to_the_workspace_log(tmp_path: Path) -> None:
    path = configure_logging(tmp_path)
    logging.getLogger("yolop_tui.test").info("diagnostic record")

    assert path == tmp_path / ".yolop" / "yolop.log"
    text = path.read_text()
    assert "diagnostic record" in text
    assert "INFO" in text


def test_run_exception_logging_persists_a_traceback(tmp_path: Path) -> None:
    path = configure_logging(tmp_path)
    try:
        raise RuntimeError("request failed")
    except RuntimeError:
        logging.getLogger("yolop_tui.app").exception("YoloP terminal run failed")

    text = path.read_text()
    assert "YoloP terminal run failed" in text
    assert "Traceback (most recent call last)" in text
    assert "RuntimeError: request failed" in text


def test_configure_logging_does_not_duplicate_handlers(tmp_path: Path) -> None:
    configure_logging(tmp_path)
    configure_logging(tmp_path)
    logging.getLogger("yolop_tui.test").warning("one record")

    text = (tmp_path / ".yolop" / "yolop.log").read_text()
    assert text.count("one record") == 1
