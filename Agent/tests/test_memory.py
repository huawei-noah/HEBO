import pytest

from agent.memory import Memory
from agent.memory import MemoryEntry


@pytest.fixture
def sample_memory():
    memory = Memory(
        [
            MemoryEntry(content="Thought 1", tags={"test", "test1"}),
            MemoryEntry(content="Thought 2", tags={"test", "test2"}),
            MemoryEntry(content="Thought 3", tags={"test2", "test3"}),
        ]
    )
    return memory


@pytest.mark.skip(reason="Needs to be fixed.")
def test_store(sample_memory):
    sample_memory.store("New Thought", {"new", "test"})
    assert sample_memory.db[0] == "New Thought"


@pytest.mark.skip(reason="Needs to be fixed.")
def test_retrieve_0(sample_memory):
    retrieved_thought = sample_memory.retrieve({"test": 1.0, "test1": 2.0})
    assert retrieved_thought == "Thought 1"

    retrieved_thought = sample_memory.retrieve({"test3": 1.0})
    assert retrieved_thought == "Thought 3"

    # Testing with weights
    retrieved_thought = sample_memory.retrieve({"test": 1.0, "test2": 3.0})
    assert retrieved_thought == "Thought 2"


@pytest.mark.skip(reason="Needs to be fixed.")
def test_retrieve_1():
    memory = Memory()

    memory.store("test1", {"system", "prompt", "random", "friendly"})
    memory.store("test2", {"user", "prompt", "alfworld", "random"})
    memory.store("test3", {"user", "prompt", "babyai"})

    assert memory.retrieve({"system": 2.0, "prompt": 1.0, "alfworld": 1.0}) == "test1"
    assert memory.retrieve({"system": -1.0, "prompt": 1.0, "alfworld": 1.0}) == "test2"
    assert memory.retrieve({"babyai": 1.0}) == "test3"
    assert memory.retrieve({"system": 2.0, "prompt": -10.0, "alfworld": 1.0}) is None


@pytest.mark.skip(reason="Needs to be fixed.")
def test_retrieve_2():
    memory = Memory()

    memory.store("test1", {"system", "prompt", "random", "friendly"})
    memory.store("test2", {"user", "prompt", "alfworld", "random"})
    memory.store("test3", {"user", "prompt", "babyai"})

    assert memory.retrieve_all({"system": 2.0, "prompt": 1.0, "alfworld": 1.0})[0] == "test1"
    assert len(memory.retrieve_all({"system": -1.0, "prompt": 1.0})) == 2
    assert len(memory.retrieve_all({"prompt": 1.0})) == 3


@pytest.mark.skip(reason="Needs to be fixed.")
def test_retrieve_all(sample_memory):
    # Adding a thought with similar tags to create a tie
    sample_memory.store("Thought 4", tags={"test", "test2"})

    retrieved_thoughts = sample_memory.retrieve_all({"test": 1.0, "test2": 2.0})
    retrieved_contents = [thought for thought in retrieved_thoughts]

    assert "Thought 2" in retrieved_contents
    assert "Thought 4" in retrieved_contents
    assert len(retrieved_thoughts) == 2


@pytest.mark.skip(reason="Needs to be fixed.")
def test_no_retrieval(sample_memory):
    retrieved_thought = sample_memory.retrieve({"nonexistent": 1.0})
    assert retrieved_thought is None

    retrieved_thoughts = sample_memory.retrieve_all({"nonexistent": 1.0})
    assert len(retrieved_thoughts) == 0


@pytest.mark.skip(reason="Needs to be fixed.")
def test_reset(sample_memory):
    sample_memory.store("Temporary Thought", {"temp", "test"})
    sample_memory.reset()

    retrieved_thought = sample_memory.retrieve({"temp": 1.0})
    assert retrieved_thought is None


@pytest.mark.skip(reason="Needs to be fixed.")
def test_empty_memory_retrieval():
    empty_memory = Memory()

    retrieved_thought = empty_memory.retrieve({"test": 1.0})
    assert retrieved_thought is None

    retrieved_thoughts = empty_memory.retrieve_all({"test": 1.0})
    assert len(retrieved_thoughts) == 0


@pytest.mark.skip(reason="Needs to be fixed.")
def test_retrieval_with_different_weights(sample_memory):
    retrieved_thought = sample_memory.retrieve({"test1": 1.0, "test2": 2.0})
    assert retrieved_thought == "Thought 3"

    retrieved_thought = sample_memory.retrieve({"test1": 2.0, "test2": 1.0})
    assert retrieved_thought == "Thought 1"


@pytest.mark.skip(reason="Needs to be fixed.")
def test_retrieval_with_same_weights(sample_memory):
    retrieved_thought = sample_memory.retrieve({"test": 1.0, "test1": 1.0, "test2": 1.0})
    assert retrieved_thought == "Thought 2"  # Because Thought 2 was inserted last


@pytest.mark.skip(reason="Needs to be fixed.")
def test_retrieval_with_no_tags(sample_memory):
    retrieved_thought = sample_memory.retrieve({})
    assert retrieved_thought is None


@pytest.mark.skip(reason="Needs to be fixed.")
def test_store_and_retrieve_new_thought(sample_memory):
    new_thought_content = "New Thought Content"
    new_tags = {"new", "tags"}

    sample_memory.store(new_thought_content, new_tags)
    retrieved_thought = sample_memory.retrieve({"new": 1.0, "tags": 1.0})

    assert retrieved_thought == new_thought_content


@pytest.mark.skip(reason="Needs to be fixed.")
def test_retrieval_with_partial_matching_tags(sample_memory):
    retrieved_thought = sample_memory.retrieve({"test": 1.0, "nonexistent": 1.0})
    assert retrieved_thought is not None  # As long as one tag matches, a thought should be retrieved


@pytest.mark.skip(reason="Needs to be fixed.")
def test_retrieval_with_no_matching_tags(sample_memory):
    retrieved_thought = sample_memory.retrieve({"nonexistent1": 1.0, "nonexistent2": 1.0})
    assert retrieved_thought is None
