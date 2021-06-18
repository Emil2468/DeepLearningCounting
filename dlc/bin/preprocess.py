import argparse

import dlc.preprocessing as preprocessing
from dlc.bin.configParser import ConfigParser
import dlc.bin.arghelper as arghelper


def get_argparser_from_class(conf):
    conf_parser = ConfigParser(conf)
    return conf_parser.get_parser()


def entry_func(args):
    conf = preprocessing.Config()
    parser = get_argparser_from_class(conf)

    parsed = parser.parse_args(args)

    arghelper.update_conf_with_parsed_args(conf, parsed)

    preprocessing.preprocess_all(conf)
