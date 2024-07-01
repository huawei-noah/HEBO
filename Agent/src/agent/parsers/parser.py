from abc import ABC
from abc import abstractmethod
from typing import Callable


class OutputParser(ABC):
    def __init__(self, expected_keys: dict[str, Callable | None]) -> None:
        super().__init__()
        self.expected_keys = expected_keys

    @abstractmethod
    def formatting_instructions(self) -> str:
        pass

    @abstractmethod
    def parse_raw_output(self, raw_output) -> dict[str, str]:
        pass
