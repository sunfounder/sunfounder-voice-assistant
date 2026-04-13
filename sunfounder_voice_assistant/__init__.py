from ._version import __version__

def ensure_tmp_dir():
    """Ensure tmp dir exists."""
    from pathlib import Path

    tmp_dir = Path("~/.tmp").expanduser()

    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True)

ensure_tmp_dir()
