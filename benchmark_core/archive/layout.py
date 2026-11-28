"""Archive layout: staging and published attempts under one archive root.

The protocol is deliberately identical for local and remote destinations so the read-only
Web can index only published attempts without knowing how they were produced.
"""
from __future__ import annotations

from pathlib import Path

from benchmark_core.inventory.config import ConfigError
from benchmark_core.workflow.state import next_attempt_id, validate_id

STAGING = "staging"
PUBLISHED = "runs"


def staging_root(archive_root):
    return Path(archive_root) / STAGING


def published_root(archive_root):
    return Path(archive_root) / PUBLISHED


def attempt_staging(archive_root, run_id, attempt_id):
    validate_id(run_id, "run_id")
    validate_id(attempt_id, "attempt_id")
    return staging_root(archive_root) / run_id / attempt_id


def attempt_published(archive_root, run_id, attempt_id):
    validate_id(run_id, "run_id")
    validate_id(attempt_id, "attempt_id")
    return published_root(archive_root) / run_id / attempt_id


def existing_attempts(archive_root, run_id):
    """Attempt ids already staged or published; ids are never reused inside one run."""
    validate_id(run_id, "run_id")
    names = set()
    for root in (staging_root(archive_root), published_root(archive_root)):
        run_dir = root / run_id
        if run_dir.is_dir():
            names |= {path.name for path in run_dir.iterdir() if path.is_dir()}
    return sorted(names)


def create_staging(archive_root, run_id, attempt_id=None):
    """Create a new staging attempt, or fail; existing attempts are never reused."""
    validate_id(run_id, "run_id")
    attempt_id = attempt_id or next_attempt_id(existing_attempts(archive_root, run_id))
    validate_id(attempt_id, "attempt_id")
    target = attempt_staging(archive_root, run_id, attempt_id)
    if target.exists():
        raise ConfigError(f"attempt is already staged: {target}")
    if attempt_published(archive_root, run_id, attempt_id).exists():
        raise ConfigError(f"attempt is already published: {attempt_id}")
    target.mkdir(parents=True, exist_ok=False)
    return target, attempt_id
