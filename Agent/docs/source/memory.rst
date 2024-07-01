.. _memory-guide:


Memory
=====================================================

Agents within the Agent framework have memory. This guide explains how this memory works and how to use it.

The memory is essentially a list and dictionary based storage system. It is used to store and retrieve information that is relevant to the agent's current state. This can be things like observations the agent has received from the environment, recent thoughts is had, or the previous actions it took.
Intrinsic functions (see guide) interact with the memory, modifying its state and retrieving information from it to perform their objectives.

The memory is made up of entries called `MemoryEntry`:code: :

.. literalinclude:: ../../src/agent/memory.py
   :pyobject: MemoryEntry

Keeping track of creation time and adding `tags`:code: enables the user to retrieve specific memory entries based on these properties. Creation time allows the agent to retrieve memory entries based on recency, while the tags enable the retrieval of a specific thought or a broader range of thoughts matching the tags.

We detail some of the most commonly used core memory methods such as storing and retrieval.

Memory store
------------------------

Any content can easily be stored in the memory by calling the `memory.store`:code: method, which takes the content and the tag(s) to assign to the memory entry:

.. code-block:: python

    def store(self, content: Any, tags: MemKey | list[MemKey]) -> None:
        """Stores a new entry in the memory.

        Parameters:
            content (Any): The main content or body of the memory entry.
            tags (MemKey | List[MemKey]): A list of tags associated with the memory entry.
        """

An example of storing an agent's observation in the memory:

.. code-block:: python

    agent.memory.store(observation, MemKey.OBSERVATION)

Memory retrieve
------------------------

The full functionality of the memory is really exposed through the retrieval functions.
The `memory.retrieve`:code: method is used to retrieve a single memory entry based on its tags and creation time:

.. code-block:: python

    def retrieve(self, tags: MemKey | list[MemKey] | dict[MemKey, float]) -> Any:
        """Retrieves the content of the best matching memory entry based on the given tags and their respective weights.
        Returns the latest memory entry if multiple memory entries have the same score. If no scores are given, the default score
        will be 1.0.

        Parameters:
            tags (MemKey | List[MemKey] | dict[MemKey, float]): Tags with optional weights to retrieve the best matching memory entry.

        Returns:
            Any: The content of the best matching memory entry or None if the max calculated weight was =< 0
        """

This method will return the best matching entry among those that have tags in common with the request.
Weights can be provided along with the tags in dictionary format to specify the importance of each tag in the retrieval process.
The sum of weights of tags present in the memory entry can then be used to select the best entry.
If multiple entries have equal score, the most recent one is returned.

Memory retrieve all
------------------------
`memory.retrieve_all`:code: works in a very similar way to `memory.retrieve`:code: , but instead of returning the single best matching entry, it returns all entries whose score is equal to the best score:

.. code-block:: python

    def retrieve_all(self, tags: MemKey | list[MemKey] | dict[MemKey, float]) -> list[Any]:
    """Retrieves the content of all memory entries that have the highest score based on the given tags
    and their respective weights.

    Parameters:
        tags (MemKey | List[MemKey] | dict[MemKey, float]): Tags with optional weights to retrieve
        the best matching memory entry.

    Returns:
        list[Any]: A list containing the content of all memory entries that have the highest matching score,
        or an empty list if that score is <=0
    """

This is useful when say you want to retrieve all past observations and actions to pass to the agent as a trajectory.


Memory keys
------------------------

A crucial element of the memory is the `MemKey`:code: enum, which defines the set of memory keys which can be used to interact with the memory.
This enum is used to define the tags that can be used for any memory operations, from storing to retrieval.
Any keys currently defined in the enum can be used.
The current state of this enum is provided below for convenience:

.. literalinclude:: ../../src/agent/memory.py
   :pyobject: MemKey

Any tags you want to give to memory entries must be defined in this enum.
This is to ensure that the memory is used consistently and without error across the framework.
As such, if you want to use a new custom tag, you must first add it to this enum.
It can then be used as desired.


Further information
------------------------

Only a subset of memory operation were presented above. For a full list of memory operations, see the `Memory`:code: class in `memory.py`:code:.
