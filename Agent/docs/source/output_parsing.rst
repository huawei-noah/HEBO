
Parsing Outputs
===============

Under `agent.parsers` we implement some interfaces to simplify parsing outputs from LLMs.
The simplest example, ``SimpleOutputParser`` class is designed to parse structured text outputs based on predefined keys.

Overview
--------

The ``SimpleOutputParser`` class extends the ``OutputParser`` class, providing methods to format instructions and parse raw outputs into a structured dictionary. It is particularly useful for parsing outputs where data is organized in a "key: value" format.

Formatting Instructions
-----------------------

The ``formatting_instructions`` method provides a clear guideline on how the output should be structured for the parser to understand. These instructions can be fed into the LLM.
It dynamically constructs instructions based on the expected keys provided during the class initialization.

Example:

.. code-block:: python

    from agent.parsers import SimpleOutputParser

    parser = SimpleOutputParser(["Thought", "Answer"])
    print(parser.formatting_instructions())

This would output:

.. code-block:: none

    Strictly answer in the format:
    Thought: <thought>
    Answer: <answer>

We recommend passing this into your jinja templates if you wish to automate the process of changing parsers (e.g., using the JsonOutputParser)

Parsing Raw Output
------------------

The ``parse_raw_output`` method takes a raw string output and extracts information based on the predefined keys. It uses regular expressions to find matches and constructs a dictionary with keys normalized to lowercase.

Example Usage:

.. code-block:: python

    raw_output = """
    Name: John Doe
    Age: 30
    Location: New York
    """

    parser = SimpleOutputParser(["Name", "Age", "Location"])
    parsed_data = parser.parse_raw_output(raw_output)
    print(parsed_data)

This would result in:

.. code-block:: none

    {
        'name': 'John Doe',
        'age': '30',
        'location': 'New York'
    }

.. note::
   When keys are repeated, SimpleOutputParser will join the values with a newline. For example an output of: `"Thought: XX\\nThought: YY"` will result in a parsed response `{"thought": "XX\\nYY"}`.

Implementation Details
----------------------

The parsing is achieved using the ``re.findall`` function from the Python standard library, with a dynamically constructed pattern based on the ``expected_keys`` provided during initialization of the parser.
This allows for flexible parsing of various output formats while maintaining a consistent dictionary structure for the results.

.. note::

    The keys in the resulting dictionary are normalized to lowercase and stripped of trailing characters like ":" and spaces for consistency.

Conclusion
----------

The ``SimpleOutputParser`` class offers a straightforward way to parse and structure raw text outputs based on predefined keys.
By following the formatting instructions and utilizing the ``parse_raw_output`` method, developers can easily extract and manipulate data from structured text outputs in their applications.
