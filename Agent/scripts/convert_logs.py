from pathlib import Path

import pyrootutils
import typer
from typing_extensions import Annotated

from agent.data_collection.log_monitoring import monitor_logs_for_conversion

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def main(
    path: Path,
    output_path: Annotated[Path, typer.Option("--output-path", "-o")] = None,
    returns_threshold: Annotated[float, typer.Option("--returns-threshold", "-r")] = 0.0001,
    use_parsed_output: Annotated[bool, typer.Option("--use-parsed-output", "-p")] = False,
    new_method: Annotated[str, typer.Option("--new-method", "-m")] = None,
    new_task: Annotated[str, typer.Option("--new-task", "-t")] = None,
    interval: Annotated[int, typer.Option("--interval", "-i")] = 30,
    deduplicate_episodes: Annotated[bool, typer.Option("--deduplicate-episodes", "-de")] = False,
    deduplicate_timesteps: Annotated[str, typer.Option("--deduplicate-timesteps", "-dt")] = "no",  # no | first | last
    chat_format: Annotated[bool, typer.Option("--chat", "-c")] = False,
    filter_initial_episodes: Annotated[bool, typer.Option("--filter-initial-episodes", "-f")] = False,
):
    """Convert logs into a dataset as they come in."""

    monitor_logs_for_conversion(
        path,
        output_path,
        returns_threshold,
        use_parsed_output,
        new_method,
        new_task,
        interval,
        deduplicate_episodes,
        deduplicate_timesteps,
        chat_format,
        filter_initial_episodes,
    )


if __name__ == "__main__":
    typer.run(main)
