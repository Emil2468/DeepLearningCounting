#   Author Emil Møller Hansen

import argparse
import importlib

from dlc.bin.arghelper import split_args, update_conf_with_parsed_args
from dlc.bin.configParser import ConfigParser


def get_argparser():
    parser = argparse.ArgumentParser(prog="train")

    parser.add_argument("trainer")

    return parser


def get_argparser_for_trainer_config(conf, prog):
    config_parser = ConfigParser(conf)
    parser = config_parser.get_parser(prog)

    return parser


def entry_func(args):
    args, help_args = split_args(args)
    parser = get_argparser()

    parsed, remaining = parser.parse_known_args(args or help_args)

    model_module = importlib.import_module("dlc.trainers." + parsed.trainer)

    conf = model_module.Config()
    model_parser = get_argparser_for_trainer_config(conf, parsed.trainer)
    model_parsed = model_parser.parse_args(help_args + remaining)

    update_conf_with_parsed_args(conf, model_parsed)

    model = model_module.Trainer(conf)
    model.run()
