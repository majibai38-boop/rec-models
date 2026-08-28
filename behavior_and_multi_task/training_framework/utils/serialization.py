"""Restricted PyTorch checkpoint loading helpers."""

from contextlib import nullcontext

import torch


DEFAULT_SAFE_GLOBALS = (getattr,)


def safe_weights_load(*args, extra_safe_globals=(), **kwargs):
    """Load tensor weights with a small, explicit weights-only allowlist.

    Recent PyTorch releases reject globals that are not allowlisted when
    ``weights_only=True``.  Older checkpoints produced around compiled models
    can reference the built-in ``getattr`` function, so it is allowed only for
    the duration of this load.
    """

    kwargs["weights_only"] = True
    allowed_globals = [*DEFAULT_SAFE_GLOBALS, *extra_safe_globals]
    safe_globals = getattr(torch.serialization, "safe_globals", None)

    if safe_globals is not None:
        get_safe_globals = getattr(
            torch.serialization, "get_safe_globals", None
        )
        existing_globals = (
            get_safe_globals() if get_safe_globals is not None else []
        )
        missing_globals = [item for item in allowed_globals if item not in existing_globals]
        context = (
            safe_globals(missing_globals)
            if missing_globals
            else nullcontext()
        )
    else:
        add_safe_globals = getattr(
            torch.serialization, "add_safe_globals", None
        )
        if add_safe_globals is not None:
            add_safe_globals(allowed_globals)
        context = nullcontext()

    with context:
        return torch.load(*args, **kwargs)
