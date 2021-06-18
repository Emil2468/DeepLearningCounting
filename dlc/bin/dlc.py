#    Author: Emil Møller Hansen

import argparse
import sys
from dlc.bin.arghelper import split_args


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("script",
                        type=str,
                        help="Name of the script to run.",
                        choices=["preprocess", "train", "evaluate", "predict"])
    return parser


def entry_func():
    args, help_args = split_args(sys.argv[1:])
    parser = get_parser()
    # parse args if there are any else try help_args
    parsed, remaining_args = parser.parse_known_args(args or help_args)

    import importlib
    script_module = importlib.import_module("dlc.bin." + parsed.script)

    # Call entry function with remaining arguments
    script_module.entry_func(remaining_args + help_args)


if __name__ == "__main__":
    entry_func()
