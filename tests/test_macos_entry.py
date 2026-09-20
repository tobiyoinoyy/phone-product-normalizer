#!/usr/bin/env python3
"""Offline static checks for the macOS designer entry point.

The test intentionally does not launch Tk, install dependencies, or process a
real photograph.  It catches packaging regressions that would otherwise leave
designers with a non-executable launcher or a GUI that cannot be imported.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
COMMAND = PROJECT / "手机图片一键处理.command"
GUI = PROJECT / "scripts" / "phone_normalizer_gui.py"
BATCH = PROJECT / "scripts" / "batch_normalize.py"


def main() -> None:
    assert COMMAND.is_file(), "macOS launcher is missing"
    assert os.access(COMMAND, os.X_OK), "macOS launcher must remain executable"
    shell_check = subprocess.run(
        ["bash", "-n", str(COMMAND)],
        capture_output=True,
        text=True,
    )
    assert shell_check.returncode == 0, shell_check.stderr
    launcher = COMMAND.read_text(encoding="utf-8")
    for marker in (
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "phone_normalizer_gui.py",
        "batch_normalize.py",
        "--output-dir",
    ):
        assert marker in launcher, f"launcher lost required marker: {marker}"

    # Importing the core before the first MPS allocation must derive a LOW
    # watermark that remains valid even when a developer supplies a custom,
    # unusually small HIGH value.
    probe = textwrap.dedent(
        """
        import os
        import scripts.segmentation
        assert os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] == '0.24'
        import torch
        torch.mps.current_allocated_memory()
        """
    )
    probe_env = dict(os.environ)
    probe_env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.3"
    probe_env.pop("PYTORCH_MPS_LOW_WATERMARK_RATIO", None)
    probe_result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(PROJECT),
        env=probe_env,
        capture_output=True,
        text=True,
    )
    assert probe_result.returncode == 0, probe_result.stderr

    # Compile/import checks are enough here; constructing a Tk root would make
    # the smoke test depend on a display server and macOS GUI session.
    compile(GUI.read_text(encoding="utf-8"), str(GUI), "exec")
    compile(BATCH.read_text(encoding="utf-8"), str(BATCH), "exec")
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from scripts.batch_normalize import STATUS_EVENTS

    assert {"scanning", "loading", "ready", "processing", "completed", "failed", "cancelled", "finished"}.issubset(
        STATUS_EVENTS
    )

    # Exercise the close-after-worker guard without creating a real Tk root.
    # This protects the race fix where a queued ``done`` event can arrive a
    # few milliseconds before the daemon thread actually returns.
    from scripts.phone_normalizer_gui import NormalizerWindow

    class FakeWorker:
        alive = True

        def is_alive(self):
            return self.alive

    class FakeRoot:
        def __init__(self):
            self.after_calls = []
            self.destroyed = False

        def after(self, delay, callback):
            self.after_calls.append((delay, callback))

        def destroy(self):
            self.destroyed = True

    window = object.__new__(NormalizerWindow)
    window._closed = False
    window.worker = FakeWorker()
    window.root = FakeRoot()
    NormalizerWindow._close_after_worker(window)
    assert window.root.after_calls and window.root.after_calls[-1][0] == 50
    window.worker.alive = False
    NormalizerWindow._close_after_worker(window)
    assert window._closed and window.root.destroyed
    print("macOS entry static smoke test passed")


if __name__ == "__main__":
    main()
