"""Model-independent runner used by the unified command-line entry point."""

from copy import deepcopy
import os
import sys

from .config import AttrDict
from .handler import ModelHandler, get_opts, get_params
from .utils.logger import logger


def run_model(
    model_factory,
    model_defaults=None,
    argv=None,
    configure_parser=None,
    configure_params=None,
    spec_loader=None,
    data_loader=None,
    test_handler_factory=None,
):
    """Parse common options and run a model through the shared framework.

    ``model_factory`` is invoked as ``model_factory(params, spec)`` only after
    the worker's rank-local device has been selected.  A framework model must
    implement the following contract:

    * ``forward(features, mode) -> dict[str, Tensor]``
    * ``loss(predictions, labels) -> scalar Tensor``
    """

    if spec_loader is None or data_loader is None or test_handler_factory is None:
        raise ValueError(
            "spec_loader, data_loader and test_handler_factory are required"
        )

    params = get_params()
    if model_defaults:
        params.update(AttrDict(deepcopy(model_defaults)))

    params = get_opts(
        sys.argv if argv is None else argv,
        params,
        configure_parser=configure_parser,
    )
    if configure_params is not None:
        params = configure_params(params)

    if params.device == "npu":
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = params.device_id
    elif params.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = params.device_id

    # Accelerator seeding is intentionally deferred until handler.run() has
    # selected this worker's LOCAL_RANK device.
    spec = spec_loader(params)
    params.dataset_name = spec.get("dataset_name", "Ali-CCP")
    logger.info(params)

    handler = ModelHandler(
        params=params,
        load_data_func=data_loader,
        test_handler=test_handler_factory(params, spec),
        model_factory=model_factory,
        spec=spec,
    )
    handler.run()
