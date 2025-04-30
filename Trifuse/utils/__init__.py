import os
import platform
import sys
from pathlib import Path
from typing import Union

MACOS, LINUX, WINDOWS = (
    platform.system() == x for x in ["Darwin", "Linux", "Windows"]
)  # environment


# Ultralytics implementation. Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py
def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)


# Ultralytics implementation. Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py
# Changes made to the original code:
# - Replaced LOGGER.warning() with print statement
# - Replace sub_dir with "Trifuse"
def get_user_config_dir(sub_dir="Trifuse"):
    """
    Return the appropriate config directory based on the environment operating system.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """
    if WINDOWS:
        path = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:  # macOS
        path = Path.home() / "Library" / "Application Support" / sub_dir
    elif LINUX:
        path = Path.home() / ".config" / sub_dir
    else:
        raise ValueError(f"Unsupported operating system: {platform.system()}")

    # GCP and AWS lambda fix, only /tmp is writeable
    if not is_dir_writeable(path.parent):
        print(
            f"user config directory '{path}' is not writeable, defaulting to '/tmp' or CWD."
            "Alternatively you can define a YOLO_CONFIG_DIR environment variable for this path."
        )
        path = (
            Path("/tmp") / sub_dir
            if is_dir_writeable("/tmp")
            else Path().cwd() / sub_dir
        )

    # Create the subdirectory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    return path


FILE = Path(__file__).resolve()
TRIFUSE_ROOT = FILE.parents[1]
USER_CONFIG_DIR = Path(get_user_config_dir())
DEFAULT_CONFIG_FILE = TRIFUSE_ROOT / "configs/default.yaml"

# DEFAULT_WEIGHT_DIR = ROOT / "weights/"

RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
