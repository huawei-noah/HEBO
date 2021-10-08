import argparse
import sys
from argparse import ArgumentParser
from ast import literal_eval
from typing import Any


def parse_dict(raw: str) -> Any:
    """
    Helper method for parsing string-encoded <dict>
    """
    try:
        pattern = raw.replace('\"', '').replace("\\'", "'")
        return literal_eval(pattern)
    except Exception as e:
        raise Exception('Failed to parse string-encoded <dict> {} with exception {}'.format(raw, e))


def parse_list(raw: str) -> Any:
    """
    Helper method for parsing string-encoded <list>
    """
    try:
        pattern = raw.replace('\"', '').replace("\\'", "'")
        return literal_eval(pattern)
    except Exception as e:
        raise Exception('Failed to parse string-encoded <list> {} with exception {}'.format(raw, e))


def get_config_from_parser(parser: ArgumentParser) -> argparse.Namespace:
    """ Read command line and store parameters in `config` Namespace"""

    config = parser.parse_args()
    config._cmdline = ' '.join(sys.argv)  # store command line for safe keeping

    return config