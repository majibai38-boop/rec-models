"""Local path checks shared by the Ali-CCP preprocessing scripts."""

from pathlib import Path


def validate_read_file(file_path):
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist or is not a file: {path}")


def validate_save_path(file_path):
    path = Path(file_path).expanduser()
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"Output path points to a directory: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {path.parent}")
