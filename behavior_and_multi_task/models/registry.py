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
from .esmm import (
    ESMM_DEFAULTS,
    add_esmm_arguments,
    build_esmm,
    configure_esmm,
)
from .esmm_torchrec_dmp import (
    ESMM_TORCHREC_DMP_DEFAULTS,
    add_esmm_torchrec_dmp_arguments,
    build_esmm_torchrec_dmp,
    configure_esmm_torchrec_dmp,
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
    "esmm": ModelRegistration(
        factory=build_esmm,
        defaults=ESMM_DEFAULTS,
        configure_parser=add_esmm_arguments,
        configure_params=configure_esmm,
    ),
    "esmm_torchrec_dmp": ModelRegistration(
        factory=build_esmm_torchrec_dmp,
        defaults=ESMM_TORCHREC_DMP_DEFAULTS,
        configure_parser=add_esmm_torchrec_dmp_arguments,
        configure_params=configure_esmm_torchrec_dmp,
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
