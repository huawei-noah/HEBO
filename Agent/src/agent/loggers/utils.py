import dataclasses
import json
from enum import Enum


class JSONEncoderV2(json.JSONEncoder):
    """ Support json encoding of dataclass objects """

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Enum):
            return super().default(o)
