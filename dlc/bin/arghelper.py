#    Author: Emil Møller Hansen


def split_args(args):
    # splits the arguments in to help args and other, because --help will be
    # parsed by the current parser, even if it should be parsed by some parser
    # in a different script
    help_args = []
    other_args = []
    for arg in args:
        if arg in ["-h", "--help"]:
            help_args.append("-h")
        else:
            other_args.append(arg)
    return other_args, help_args


def update_conf_with_parsed_args(conf, parsed):
    """
        Updates the conf object in-place with the updated parametes from parsed
    """
    for arg in vars(parsed):
        val = getattr(parsed, arg)
        setattr(conf, arg, val)
