#    Author: Emil Møller Hansen

import argparse
from dlc.bin import arghelper
from dlc.bin.configParser import ConfigParser
import importlib


def get_argparser():
    parser = argparse.ArgumentParser(prog="predict")

    parser.add_argument("trainer")

    return parser


def get_argparser_for_trainer_config(conf, prog):
    config_parser = ConfigParser(conf)
    parser = config_parser.get_parser(prog)

    return parser


def entry_func(args):
    args, help_args = arghelper.split_args(args)
    parser = get_argparser()

    parsed, remaining = parser.parse_known_args(args or help_args)

    model_module = importlib.import_module("dlc.trainers." + parsed.trainer)

    conf = model_module.Config()
    model_parser = get_argparser_for_trainer_config(conf, parsed.trainer)
    model_parsed = model_parser.parse_args(help_args + remaining)

    arghelper.update_conf_with_parsed_args(conf, model_parsed)
    model = model_module.Trainer(conf)
    model.predict()
