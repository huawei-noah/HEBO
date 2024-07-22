# Agent

This branch is under development, but is expected to work; please open an issue if you encounter any problems.

To merge your changes on main, please open a PR.

For the code replicating our paper's experiments please switch to the `dev-release` branch.

## Latest Changes

- 2024-03-01 Reworked memory. Memory operations will now require valid `MemKey`s as tags, and memory retrievals will return content directly (no need for `.content`). Docs can be found [here](docs/source/memory.rst).
- 2024-02-29 Added Human (stdin) as a language backend. Use by setting `llm@agent.llm=human`.
- 2024-02-22 Reworked flows. Docs can be found [here](docs/source/flows.rst).
- 2024-02-22 Reworked intrinsic functions. Docs can be found [here](docs/source/intrinsic_functions.rst).
- 2024-02-20 Implement a LoopFlow to allow running loops of commands or sub-flows.
- 2024-02-15 Restructures LLM interface under models/
- 2024-01-27 *Major* Changes the way we support multi-agent. This means that all configs are cleaner and just define a single agent.
- 2024-01-20 Update vLLM launches script

## Documentation

Quick installation instructions can be found under [here](docs/source/installation.rst).

Then you can compile the documentation under `docs/` by running:

```bash
cd docs
make html
```
