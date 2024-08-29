import enum
import itertools
import json
import time
from hashlib import sha256
from typing import TypedDict

import rich
import rich.tree
from omegaconf import DictConfig
from omegaconf import OmegaConf
from rich.markup import escape

# Cost per token
_BASE_INPUT_LLM_COST = 0.03 / 1000
_BASE_OUTPUT_LLM_COST = 0.06 / 1000

_BASE_LLM_PARAMS = int(1.7 * 1e12)


class Message(TypedDict):
    role: str
    content: str


class Tag(str, enum.Enum):
    EPISODE = "episode"
    TIMESTEP = "timestep"
    TIMESTAMP = "timestamp"
    DONE = "done"
    REWARD = "reward"
    ERROR = "error"
    LLM = "llm"

    @staticmethod
    def universal_tags(episode: int = None, timestep: int = None, timestamp: float = None):
        return {
            Tag.EPISODE: episode,
            Tag.TIMESTEP: timestep,
            Tag.TIMESTAMP: timestamp if timestamp is not None else time.time(),
        }


class Logger:
    def __init__(self, project_name, project_cfg: DictConfig):
        self._logs = []

        self.config_hash = sha256(
            json.dumps(
                {
                    k: v
                    for k, v in OmegaConf.to_container(project_cfg).items()
                    if k not in ["seed", "extras", "paths", "tags"]
                },
                sort_keys=True,
            ).encode("utf8")
        ).hexdigest()[-10:]

        self._start_time = time.time()
        self._prev_time = None
        self._cur_timestep = 0
        self._cur_episode = 0

    def update_state(self, *args, **kwargs):
        pass

    def log_metrics(
        self,
        data: dict[str, str | float | int | list[Message]],
        episode: int | None = None,
        timestep: int | None = None,
    ):
        assert episode is None or episode >= self._cur_episode
        assert timestep is None or (timestep >= self._cur_timestep or episode > self._cur_episode)

        self._cur_timestep = self._cur_timestep if timestep is None else timestep
        self._cur_episode = self._cur_episode if episode is None else episode

        universal_tags = Tag.universal_tags(self._cur_episode, self._cur_timestep)

        for key, value in data.items():
            self._logs.append({**universal_tags, key: value})

    def save_metrics(self):
        self._logs = []
        pass


class ManyLoggers(Logger):
    def __init__(self, loggers: list[Logger]):
        self.loggers = loggers

    def update_state(self, *args, **kwargs):
        for logger in self.loggers:
            logger.update_state(*args, **kwargs)

    def log_metrics(self, *args, **kwargs):
        for logger in self.loggers:
            logger.log_metrics(*args, **kwargs)

    def save_metrics(self):
        for logger in self.loggers:
            logger.save_metrics()


class StdoutLogger(Logger):
    def save_metrics(self):
        tree = rich.tree.Tree("Logs")
        for log in self._logs:
            keys = set(log.keys()) - set(Tag.universal_tags().keys())
            for key in keys:
                tree.add(key).add(escape(str(log[key])))

        rich.print(tree)
        self._logs = []


class FileSystemLogger(Logger):
    def __init__(self, project_cfg: DictConfig, save_to_file: str, **kwargs):
        super().__init__(project_cfg=project_cfg, **kwargs)
        self.output_dir = project_cfg.paths.output_dir
        self.save_to_file = save_to_file

    def save_metrics(self):
        with open(self.output_dir + "/" + self.save_to_file, "a") as outfile:
            for record in self._logs:
                json.dump(record, outfile)
                outfile.write("\n")
        self._logs = []


class APIUsageLogger(Logger):
    def __init__(self, project_cfg: DictConfig, **kwargs):
        super().__init__(project_cfg=project_cfg, **kwargs)
        self.input_usage = 0
        self.output_usage = 0

    def update_state(self, llm_num_params: int):
        self.llm_num_params = llm_num_params
        print(f"llm_num_params: {self.llm_num_params}")

    def log_metrics(
        self,
        data: dict[str, str | float | int | list[Message]],
        episode: int | None = None,
        timestep: int | None = None,
    ):
        for key, value in data.items():
            if key.startswith("api_usage"):
                self._logs.append({key: value})
            if key.startswith("api_usage:input"):
                self.input_usage += value
            elif key.startswith("api_usage:output"):
                self.output_usage += value

    def save_metrics(self):
        print(self._logs)
        self._logs = []

    def reset_usage_metrics(self):
        self.input_usage = 0
        self.output_usage = 0

    def get_cost(self):
        return ((self.input_usage * _BASE_INPUT_LLM_COST) + (self.output_usage * _BASE_OUTPUT_LLM_COST)) * (
            self.llm_num_params / _BASE_LLM_PARAMS
        )


class TrainingDataLogger(Logger):
    parsed = True

    def __init__(self, *args, only_successes=False, skip_tags=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.only_successes = only_successes
        self.skip_tags = set() if skip_tags is None else set(skip_tags)

    @property
    def _logs_by_episode(self):
        return dict(itertools.groupby(self._logs, lambda line: line[Tag.EPISODE]))

    def make_episode_dict(self, episode_log):
        history = []
        episode_dict = {}
        assistant_role, removed_role = ("parsed_output", "output") if self.parsed else ("output", "parsed_output")

        for line in episode_log:
            for k, v in line.items():
                if k.startswith(Tag.LLM):
                    role = k.split(":")[-1]
                    if role == removed_role:
                        continue
                    role = "assistant" if role == assistant_role else role
                    history.append({"role": role, "content": v})
                elif not any(k.startswith(skip) for skip in self.skip_tags):
                    episode_dict[k] = v

        return episode_dict

    def should_log_episode(self, episode_dict):
        return not self.only_successes or (
            not isinstance(episode_dict[Tag.REWARD], str) and episode_dict[Tag.REWARD] > 0
        )

    def save_metrics(self):
        done_log = self._logs[-1]

        if not done_log.get(Tag.DONE):
            self._logs = []
            return

        episode_dicts = [self.make_episode_dict(ep_log) for ep_log in self._logs_by_episode.values()]

        self._logs = list(filter(self.should_log_episode, episode_dicts))
        self._logs = []


class FakeLogger(Logger):
    def __init__(self):
        pass

    def log_metrics(
        self,
        data: dict[str, str | float | int | list[Message]],
        episode: int | None = None,
        timestep: int | None = None,
    ):
        pass

    def save_metrics(self):
        pass
