import os
from pathlib import Path
from typing import Dict, Union, Any
import subprocess

import pandas as pd


def _download_file_safely(url: str, filepath: Path, manual_url: str) -> None:
    """
    Safely downloads a file using a temporary file approach.

    Args:
        url: Download URL
        filepath: Final destination path
        manual_url: URL for manual download in case of failure
    """
    import tempfile
    import shutil
    import os

    # Create a temporary file for download
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Download to temporary location
        subprocess.run(
            f"wget -O '{temp_path}' '{url}'",
            shell=True,
            capture_output=True,
        )

        # Check if download was successful
        temp_file_stat = os.stat(temp_path)
        if temp_file_stat.st_size <= 0:
            raise ValueError("Downloaded file has zero size")

        # Move to final location if successful
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        shutil.move(temp_path, filepath)

    except Exception as e:
        # Clean up temporary file in case of failure
        if os.path.exists(temp_path):
            os.unlink(temp_path)

        error_msg = (
            f"Failed to download file. "
            f"Please download it manually from: {manual_url}"
        )
        raise FileNotFoundError(error_msg) from e



def clean_directory_path(
    cache_dir: Union[str, Path, None], default_dir: str = "data"
) -> Path:
    """
    Ensure a directory path exists, creating it if necessary.

    Args:
        cache_dir: Directory path to clean/create
        default_dir: Default directory name if cache_dir is None

    Returns:
        Path object to the clean directory
    """
    if cache_dir is None:
        cache_dir = Path(os.getcwd()) / default_dir
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        os.makedirs(cache_dir)
    return cache_dir


def save_dict(data: Dict[str, Any], output_dir: Path) -> None:
    """
    Recursively save a dictionary structure to disk.

    Args:
        data: Dictionary to save
        output_dir: Directory path where to save data
    """
    assert isinstance(data, dict)
    if not output_dir.is_dir():
        os.makedirs(output_dir)
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            v.to_parquet(output_dir / f"{k}.parquet")
        else:
            save_dict(v, output_dir=output_dir / k)


def load_dict(input_dir: Path) -> Dict[str, Any]:
    """
    Recursively load a dictionary structure from disk.

    Args:
        input_dir: Directory path from which to load data

    Returns:
        Dictionary with loaded data
    """
    data = {}
    for o in os.scandir(input_dir):
        if o.name.endswith(".parquet"):
            k = o.name.replace(".parquet", "")
            data[k] = pd.read_parquet(o)
        elif o.is_dir():
            data[o.name] = load_dict(o)
    return data
