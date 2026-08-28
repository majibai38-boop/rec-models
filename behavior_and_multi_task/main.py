"""Unified command-line entry point for behavior and multi-task models."""

import argparse
import os
import sys

from behavior_and_multi_task.training_framework import run_model
from behavior_and_multi_task.data_process.aliccp import (
    TestAliccpHandler,
    get_spec,
    load_data,
)
from behavior_and_multi_task.models.registry import (
    get_model_registration,
    list_models,
)


def select_model(argv):
    """Select the model before constructing the model-specific full parser."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--model",
        choices=list_models(),
        default=os.environ.get("MODEL", "eta"),
    )
    options, _ = parser.parse_known_args(argv[1:])
    return options.model


def main(argv=None):
    argv = sys.argv if argv is None else argv
    model_name = select_model(argv)
    registration = get_model_registration(model_name)

    def configure_parser(parser):
        parser.add_argument(
            "--model",
            choices=list_models(),
            default=model_name,
            help="model registered in behavior_and_multi_task.models",
        )
        if registration.configure_parser is not None:
            parser = registration.configure_parser(parser)
        return parser

    run_model(
        registration.factory,
        model_defaults=registration.defaults,
        argv=argv,
        configure_parser=configure_parser,
        configure_params=registration.configure_params,
        spec_loader=get_spec,
        data_loader=load_data,
        test_handler_factory=TestAliccpHandler,
    )


if __name__ == "__main__":
    main()
