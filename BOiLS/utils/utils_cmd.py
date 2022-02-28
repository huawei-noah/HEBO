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
