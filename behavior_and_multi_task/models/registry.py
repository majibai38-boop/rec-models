"""Model registry used by the unified entry point."""

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from .autoint import (
    AUTOINT_DEFAULTS,
    add_autoint_arguments,
    build_autoint,
    configure_autoint,
)
from .dien import (
    DIEN_DEFAULTS,
    add_dien_arguments,
    build_dien,
    configure_dien,
)
from .essm import (
    ESSM_DEFAULTS,
    add_essm_arguments,
    build_essm,
    configure_essm,
)
from .eta import (
    ETA_DEFAULTS,
    add_eta_arguments,
    build_eta,
    configure_eta,
)
from .eta_legacy import ETA_LEGACY_DEFAULTS, build_eta_legacy


@dataclass(frozen=True)
class ModelRegistration:
    factory: Callable
    defaults: Mapping
    configure_parser: Optional[Callable] = None
    configure_params: Optional[Callable] = None


MODEL_REGISTRY = {
    "autoint": ModelRegistration(
        factory=build_autoint,
        defaults=AUTOINT_DEFAULTS,
        configure_parser=add_autoint_arguments,
        configure_params=configure_autoint,
    ),
    "dien": ModelRegistration(
        factory=build_dien,
        defaults=DIEN_DEFAULTS,
        configure_parser=add_dien_arguments,
        configure_params=configure_dien,
    ),
    "eta": ModelRegistration(
        factory=build_eta,
        defaults=ETA_DEFAULTS,
        configure_parser=add_eta_arguments,
        configure_params=configure_eta,
    ),
    "eta_legacy": ModelRegistration(
        factory=build_eta_legacy,
        defaults=ETA_LEGACY_DEFAULTS,
        configure_parser=add_eta_arguments,
        configure_params=configure_eta,
    ),
    "essm": ModelRegistration(
        factory=build_essm,
        defaults=ESSM_DEFAULTS,
        configure_parser=add_essm_arguments,
        configure_params=configure_essm,
    ),
}


def list_models():
    return tuple(sorted(MODEL_REGISTRY))


def get_model_registration(name):
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model {name!r}; available models: {list_models()}"
        ) from exc
