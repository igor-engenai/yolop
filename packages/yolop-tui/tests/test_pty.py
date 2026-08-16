import os
import pty
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Unix pseudo-terminal test")


def test_cli_runs_inline_and_ctrl_c_exits_in_a_real_pseudo_terminal(tmp_path: Path) -> None:
    master, slave = pty.openpty()
    environment = {**os.environ, "TERM": "xterm-256color"}
    process = subprocess.Popen(
        [sys.executable, "-m", "yolop_tui.cli"],
        cwd=tmp_path,
        env=environment,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    os.close(slave)
    output = bytearray()
    try:
        deadline = time.monotonic() + 5
        while b"prompt" not in output and time.monotonic() < deadline:
            readable, _, _ = select.select([master], [], [], 0.1)
            if readable:
                output.extend(os.read(master, 65536))
        assert b"prompt" in output

        os.write(master, b"\x03")
        assert process.wait(timeout=5) == 0
        while True:
            readable, _, _ = select.select([master], [], [], 0)
            if not readable:
                break
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)

    assert b"\x1b[?1049h" not in output
    assert (tmp_path / ".yolop" / "runtime.db").is_file()
