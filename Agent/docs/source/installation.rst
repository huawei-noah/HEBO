.. _installation-guide:

Installation Guide
==================

This document provides detailed instructions for installing and setting up **Agent**. Follow these steps to ensure a smooth installation process.

Cloning the Repository
----------------------

Begin by cloning the **Agent** repository and navigating to the project directory:

1. Clone the repository:

   .. code-block:: bash

       git clone [Repository URL]

2. Change directory to the cloned repository:

   .. code-block:: bash

       cd agent

Installing Conda
----------------

Next, install Conda. You can choose any preferred method for this, but we recommend the microconda distribution. For detailed instructions, please visit the `official Conda installation guide <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

Creating and Activating the Environment
---------------------------------------

After installing Conda, create and activate the **agent** environment:

1. To create the environment, run:

   .. code-block:: bash

       conda create -n agent --file conda-linux-64.lock

2. To activate the environment, execute:

   .. code-block:: bash

       conda activate agent

Installing Dependencies
-----------------------

With the environment active, proceed to install the dependencies:

.. code-block:: bash

    poetry install

Optionally, you can install all extra dependencies:

.. code-block:: bash

    poetry install --all-extras

Note: if the poetry install fails, run multiple times until successful. Failing this it is recommended that you check your proxy/internet restrictions.
You can also export poetry's dependencies to a `requirements.txt` format by running:

.. code-block:: bash

    poetry export --without-hashes --format=requirements.txt > requirements.txt

and then installing with

.. code-block:: bash

    pip install -r requirements.txt


Explanation of Extras
~~~~~~~~~~~~~~~~~~~~~

The extra dependencies are categorized as follows:

- **backend**: For running LLM servers.
- **training**: For RL or SFT training.

Extras for different environments include:

- **alfworld**
- **babyai**
- **humaneval**
- **webshop**

These can be installed separately if not included in `--all-extras`. For example, to install only the **alfworld** extras, run:

.. code-block:: bash

    poetry install --extras alfworld
