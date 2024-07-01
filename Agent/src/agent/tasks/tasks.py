from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict


class DatasetOutOfBoundsException(Exception):
    """Raised when dataset index is greater than dataset length."""

    pass


class ActionSpace(Enum):
    DISCRETE = 1
    CONTINUOUS = 2


class Task(ABC):
    def __init__(
        self,
        **kwargs,
    ):
        self.id = None
        self.action_space = NotImplemented

    @abstractmethod
    def step(self, action) -> tuple[Dict[str, Any], float, bool]:
        """Perform an action and return the next observation, reward, and done."""
        pass

    @abstractmethod
    def reset(self, next_subtask: str | None) -> Dict[str, Any]:
        """Reset the environment and return the initial observation."""
        pass

    @abstractmethod
    def answer_parser(self, raw_response: str) -> str:
        """Return a parsed response."""
        pass
