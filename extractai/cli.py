from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _resolve_streamlit_app_path() -> Path:
    return Path(__file__).resolve().with_name("app.py")


def run_app(*, extra_args: Sequence[str] | None = None) -> int:
    if importlib.util.find_spec("streamlit") is None:
        print(
            "Streamlit is not installed. Install demo extras first: "
            "python -m pip install \"extractai[demo]\"",
            file=sys.stderr,
        )
        return 1

    args = list(extra_args or [])
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_resolve_streamlit_app_path()),
        *args,
    ]
    return subprocess.call(command)


def main() -> None:
    raise SystemExit(run_app(extra_args=sys.argv[1:]))
