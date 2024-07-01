from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from agent.utils import pylogger

log = pylogger.get_pylogger(__name__)


class MemKey(str, Enum):
    """An enumeration to represent the keys used in the memory."""

    # Generic keys
    AVAILABLE_ACTIONS = "_available_actions"  # Available actions for the agent to choose from
    EXTERNAL_ACTION = "_action"  # External actions the agent has taken
    NEXT_PLANNED_ACTION = "_next_planned_action"  # Next action the agent plans to take
    OBSERVATION = "_text_obs"  # Observations the agent has observed from the task
    RAW_ACTION_OUTPUT = "_raw_action_output"  # Raw outputs from the LLM
    REWARD = "_reward"  # Reward received from the environment by the agent at each timestep
    TASK_CATEGORY = "_task_category"  # Category of the task (only for some environments)
    THOUGHT = "_thought"  # Thoughts produced by the agent

    # Reflect keys
    REFLECTION = "_reflection"  # Reflections produced by the agent

    # Least-to-most keys
    SUBPROBLEM = "_subproblem"  # Subproblems produced by the agent

    # Self-consistency keys
    NEXT_PLANNED_ACTION_DIVERSE = "_next_possible_action_diverse"  # Diverse set of next possible actions (for SC)

    # # Swift-Sage keys
    # NEXT_PLANNED_ACTION_BUFFER = "_next_planned_action_buffer"  # Buffer of next actions to take (for SwiftSage)

    # RAG keys
    DB_QUERY = "_db_query"
    RAG_RETRIEVAL = "_rag_retrieval"

    # Google search keys
    GOOGLE_QUERY = "_google_query"  # Query used for google search
    LINK_RETRIEVED = "_link_retrieved"  # Link retrieved from google search
    NUM_CHUNKS = "_num_chunks"  # Number of chunks to summarize google search
    SNIPPET_RETRIEVED = "_snippet_retrieved"  # Snippet retrieved from google search
    SUMMARIZED_TEXT = "_summarized_text"  # Summarized text from google search
    TEXT_RETRIEVED = "_text_retrieved"  # Text retrieved from google search
    TITLE_RETRIEVED = "_title_retrieved"  # Title retrieved from google search

    # Trajectory keys
    FULL_EPS_REWARDS = "_full_eps_rewards"  # Full list of rewards of the agent for the current episode
    FULL_EPS_TRAJECTORY = "_full_eps_trajectory"  # Full trajectory of the agent for the current episode
    TRAJECTORY = "_trajectory"  # Agent trajectory by timestep


class MemoryEntry(BaseModel):
    """A class to represent a memory entry with creation time, content, and associated tags.

    Attributes:
        created_at (datetime): The time when the memory entry is created. Default is the current time.
        content (Any): The main content or body of the memory entry.
        tags (Union[Set[MemKey], List[MemKey]]): A set or list of tags associated with the memory entry.
    """

    created_at: datetime = datetime.now()
    content: Any
    tags: set[MemKey]


class Memory:
    """A class to manage the storing and retrieving of memory entries based on their associated
    tags and weights.

    Attributes:
        db (list[MemoryEntry]): A database(list) to store memory entries.
    """

    def __init__(self, initial_memories: list[MemoryEntry] | None = None):
        """Initializes the memory with an optional list of memory entries.

        Parameters:
            initial_memories (list[MemoryEntry] | None): An optional list of memory entries to pre-populate the memory.
        """
        if initial_memories is None:
            initial_memories = []

        self.db: list[MemoryEntry] = []
        self._initial_memories = initial_memories
        self.mem_keys = MemKey
        self.history_summary = None
        self.logger = None
        self.reset()

    def __contains__(self, tags: MemKey | list[MemKey]) -> bool:
        """Checks if the memory contains any memory entry with the given tags.

        Parameters:
            tags (MemKey | List[MemKey]): A list of tags to check for.

        Returns:
            bool: True if the memory contains any memory entry with the given tags, False otherwise.
        """

        for entry in self.db:
            if entry.tags.issuperset(set(tags)):
                return True
        return False

    def add_tags(self, ref_tag: MemKey, new_tags: MemKey | list[MemKey]) -> None:
        """Adds new tags to all instances of the reference tag in existing memory entries.

        Parameters:
            ref_tag (MemKey): The reference tag to which the new tags will be added.
            new_tags (MemKey | List[MemKey]): A list of new tags to be added.
        """

        new_tags = set(new_tags)

        assert ref_tag in set(self.mem_keys) and new_tags.issubset(set(self.mem_keys)), (
            "The reference tag and the new tags must be valid memory keys (MemKey enum). "
            "Please add them to the memory keys enum if you are intentionally creating new keys."
        )

        for entry in self.db:
            if ref_tag in entry.tags:
                entry.tags = entry.tags.union(new_tags)

    def delete(self, tags: MemKey | list[MemKey]) -> None:
        """Deletes all instances of the tags from existing memory entries. If any memory entry has
        no tags, delete it.

        Parameters:
            tags (MemKey | List[MemKey]): A list of tags associated with the memory entries to be deleted.
        """

        tags = set(tags)

        assert tags.issubset(
            set(self.mem_keys)
        ), "All deleted tags must be valid memory keys (MemKey enum). Please double check your tags and try again."

        for entry in self.db:
            entry.tags = entry.tags - tags
            if len(entry.tags) == 0:
                self.db.remove(entry)

    def rename(self, old_tag: MemKey, new_tag: MemKey) -> None:
        """Renames all instances of the old tag to the new tag in existing memory entries.

        Parameters:
            old_tag (MemKey): The old tag to be renamed.
            new_tag (MemKey): The new tag to replace the old tag.
        """

        assert old_tag in set(self.mem_keys) and new_tag in set(self.mem_keys), (
            "The old and new tags must be valid memory keys (MemKey enum). Please double check your tags, "
            "and add the new_tag to the memory keys enum if you are intentionally creating a new key."
        )

        for entry in self.db:
            if old_tag in entry.tags:
                entry.tags.remove(old_tag)
                entry.tags.add(new_tag)

    def reset(self) -> None:
        """Resets the memory to its initial state (the memories with which it was initialised)."""

        self.db = []
        for mem in self._initial_memories:
            self.store(mem.content, mem.tags)

    def retrieve(self, tags: MemKey | list[MemKey] | dict[MemKey, float]) -> Any:
        """Retrieves the content of the best matching memory entry based on the given tags and
        their respective weights. Returns the latest memory entry if multiple memory entries have
        the same score. If no scores are given, the default score will be 1.0.

        Parameters:
            tags (MemKey | List[MemKey] | dict[MemKey, float]): Tags with optional weights to retrieve
            the best matching memory entry.

        Returns:
            Any: The content of the best matching memory entry or None if the max calculated weight was =< 0
        """

        if not isinstance(tags, dict):
            tags = {tag: 1.0 for tag in set(tags)}

        assert set(tags.keys()).issubset(
            set(self.mem_keys)
        ), "All retrieved tags must be valid memory keys (MemKey enum). Please double check your tags and try again."

        best, best_score = None, float("-inf")

        for entry in self.db:
            score = sum([value for key, value in tags.items() if key in entry.tags])
            if score > best_score and score > 0:
                best = entry.content
                best_score = score

        return best

    def retrieve_all(self, tags: MemKey | list[MemKey] | dict[MemKey, float]) -> list[Any]:
        """Retrieves the content of all memory entries that have the highest score based on the
        given tags and their respective weights.

        Parameters:
            tags (MemKey | List[MemKey] | dict[MemKey, float]): Tags with optional weights to retrieve
            the best matching memory entry.

        Returns:
            list[Any]: A list containing the content of all memory entries that have the highest matching score,
            or an empty list if that score is <=0
        """

        if not isinstance(tags, dict):
            tags = {tag: 1.0 for tag in set(tags)}

        assert set(tags.keys()).issubset(
            set(self.mem_keys)
        ), "All retrieved tags must be valid memory keys (MemKey enum). Please double check your tags and try again."

        best, best_score = [], float("-inf")

        for entry in self.db:
            score = sum([value for key, value in tags.items() if key in entry.tags])
            if score > best_score and score > 0:
                best = [entry.content]
                best_score = score
            elif score == best_score:
                best.append(entry.content)

        return best

    def store(self, content: Any, tags: MemKey | list[MemKey]) -> None:
        """Stores a new entry in the memory.

        Parameters:
            content (Any): The main content or body of the memory entry.
            tags (MemKey | List[MemKey]): A list of tags associated with the memory entry.
        """

        tags = set(tags)

        assert tags.issubset(set(self.mem_keys)), (
            "All stored tags must be valid memory keys (MemKey enum). If you are intentionally creating new keys, "
            "please add them to the memory keys enum."
        )

        if self.logger:
            self.logger.log_metrics({f"memory:store:{'|'.join(tags)}": content})

        # By inserting at the beginning, newer memory entries will get priority
        self.db.insert(0, MemoryEntry(content=content, tags=tags))


if __name__ == "__main__":
    memory = Memory()

    memory.store("This is a random system prompt.", {"system", "prompt", "random", "friendly"})
    memory.store("This is a random user prompt for alfworld.", {"user", "prompt", "alfworld", "random"})
    memory.store("This is a random user prompt for babyai.", {"user", "prompt", "babyai"})

    print(memory.retrieve({"system": 2.0, "prompt": 1.0, "alfworld": 1.0}))
