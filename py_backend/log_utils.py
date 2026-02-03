import os
from pathlib import Path
from datetime import datetime

_LOG_FILE_HANDLE = None


def _logs_root() -> Path:
    return Path(__file__).resolve().parent.parent / "logs"


def _daily_log_dir(now: datetime) -> Path:
    day_dir = _logs_root() / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir


def new_log_path(prefix: str, now: datetime | None = None) -> Path:
    if now is None:
        now = datetime.now()
    ts = now.strftime("%H-%M-%S")
    pid = os.getpid()
    return _daily_log_dir(now) / f"{prefix}-{ts}-{pid}.log"


def env_log_path(prefix: str) -> Path | None:
    key = f"IRIS_{prefix.upper()}_LOG_PATH"
    override = os.environ.get(key)
    if override:
        return Path(override)
    return None


def setup_process_logging(prefix: str) -> Path:
    """Redirect stdout/stderr to a timestamped log file under logs/YYYY-MM-DD/."""
    global _LOG_FILE_HANDLE

    override_path = os.environ.get("IRIS_LOG_PATH")
    log_path = Path(override_path) if override_path else new_log_path(prefix)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = open(log_path, "a", buffering=1)
    os.dup2(log_file.fileno(), 1)
    os.dup2(log_file.fileno(), 2)

    _LOG_FILE_HANDLE = log_file

    os.environ["IRIS_LOG_PATH"] = str(log_path)
    os.environ["IRIS_LOG_DIR"] = str(log_path.parent)
    os.environ["IRIS_LOG_PREFIX"] = prefix

    return log_path
