from dataclasses import dataclass, field
import os
from pathlib import Path
import secrets


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("PROFILING_WEB_DATA", "profiling_web/data")).resolve())
    token: str = field(default_factory=lambda: os.environ.get("PROFILING_WEB_TOKEN", ""))
    max_upload_bytes: int = field(default_factory=lambda: int(os.environ.get("PROFILING_WEB_UPLOAD_MB", "1024")) * 1024**2)
    storage_bytes: int = field(default_factory=lambda: int(os.environ.get("PROFILING_WEB_STORAGE_MB", "20480")) * 1024**2)
    memory_mb: int = field(default_factory=lambda: int(os.environ.get("PROFILING_WEB_MEMORY_MB", "8192")))
    timeout_seconds: int = field(default_factory=lambda: int(os.environ.get("PROFILING_WEB_TIMEOUT", "1800")))
    allowed_hosts: list = field(default_factory=lambda: os.environ.get("PROFILING_WEB_HOSTS", "localhost,127.0.0.1,[::1]").split(","))
    secure_cookie: bool = field(default_factory=lambda: os.environ.get("PROFILING_WEB_HTTPS", "0") == "1")

    def prepare(self):
        self.data_dir = Path(self.data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.data_dir / "jobs").mkdir(exist_ok=True, mode=0o700)
        if min(self.max_upload_bytes, self.storage_bytes, self.memory_mb, self.timeout_seconds) <= 0:
            raise ValueError("Resource limits must be positive")
        if not self.token:
            secret = self.data_dir / "access-token"
            try:
                fd = os.open(secret, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                self.token = secret.read_text(encoding="utf-8").strip()
            else:
                self.token = secrets.token_urlsafe(32)
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(self.token + "\n")
        if len(self.token) < 24:
            raise ValueError("PROFILING_WEB_TOKEN must contain at least 24 characters")
        return self


INPUTS = {"gpu_trace": "gpu.json", "npu_trace": "npu.json", "npu_kernel_details": "kernel_details.csv"}


def disk_usage(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file() and not p.is_symlink())
