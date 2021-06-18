#    Author: Emil Møller Hansen

import argparse
from typing import List, Union, get_type_hints
import numpy as np

# Assumptions:
# Lists are of any length but fixed types.
# Tuples can contain any mixed types but are of fixed length.
# Parser only handles numeric types, bools, str, lists, tuples, np.ndarray,
# Union/Optional types


class ConfigParser():

    def __init__(self, conf):
        self.parameters = {
            key: val for key, val in vars(conf).items() if not key.startswith("_")
        }

        self.annotations = get_type_hints(type(conf))
        self.parser = None

    def get_parser(self, prog=None):
        if self.parser is None:
            self.parser = argparse.ArgumentParser(prog=prog)

            for (key, val) in self.parameters.items():
                t = self.annotations[key]
                try:
                    # in python3.6 you get List, in python3.7 you get list
                    if t.__origin__ == list or t.__origin__ == List:
                        nargs = "*"
                        # get the type of list to ensure each element has correct type
                        t = t.__args__[0]
                    elif t.__origin__ == tuple:
                        nargs = len(val)
                        types = t.__args__
                        action = convert_tuple_args_to_type(types)
                        # start by parsing all arguments as str, then they
                        # will be converted in the action
                        t = str
                    elif t.__origin__ == Union:
                        nargs = None
                        action = parse_optional(t)
                        t = str
                except AttributeError:  # handler for types with no origin
                    if t == np.ndarray:
                        # since argparse will always return list,
                        # convert to np.ndarray here
                        action = convert_args_to_type(np.ndarray)
                        if len(val) > 0:
                            t = self.get_safe_type(val[0])
                        else:
                            t = str
                        nargs = "*"
                    else:
                        nargs = None
                        action = "store"
                # bool is a special case because bool("False") -> True,
                # so special handling is needed
                if t == bool:
                    t = str2bool(key)
                self.parser.add_argument("--" + key,
                                         default=val,
                                         type=t,
                                         nargs=nargs,
                                         action=action,
                                         help=f"Default: {val}")
        return self.parser

    # returns the type of val, unless val is NoneType, then returns str
    def get_safe_type(_, val):
        return type(val) if not isinstance(val, type(None)) else str


def convert_args_to_type(t):

    class ArgumentTypeConvertor(argparse.Action):

        def __call__(self, parser, namespace, values, option_string):
            # this is not as generic as I would have liked since there
            # is the special case with ndarrays
            if t == np.ndarray:
                setattr(namespace, self.dest, np.array(values))
            else:
                setattr(namespace, self.dest, t(values))

    return ArgumentTypeConvertor


def convert_tuple_args_to_type(element_types):

    class TupleElementTypeConvertor(argparse.Action):

        def __call__(self, parser, namespace, values, option_string):
            result = []
            for (val, t) in zip(values, element_types):
                if t == bool:
                    result.append(str2bool(self.dest)(val))
                else:
                    result.append(t(val))
            setattr(namespace, self.dest, tuple(result))

    return TupleElementTypeConvertor


def str2bool(arg_name):

    def str2bool_(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError(
                f'Boolean value expected for argument {arg_name}.')

    return str2bool_


def parse_optional(t):

    class OptionalArgumentParser(argparse.Action):

        def __call__(self, parser, namespace, value, option_string):
            if value == "None":
                setattr(namespace, self.dest, None)
            else:
                main_type = t.__args__[0]
                if main_type == bool:
                    main_type = str2bool(self.dest)
                setattr(namespace, self.dest, main_type(value))

    return OptionalArgumentParser
