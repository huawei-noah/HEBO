import random

import mlrq
import torch
from torch.utils.data import IterableDataset


@mlrq.distribute(
    priority="high",
)
def add_trajectory(replay_buffer, trajectory):
    replay_buffer.add_trajectories(trajectory)


class ReplayBuffer(IterableDataset):
    def __init__(self, on_policy: bool, dataset_size: int, update_frequency: int):
        super(ReplayBuffer).__init__()
        self.on_policy = on_policy
        self.dataset_size = dataset_size
        self.update_frequency = update_frequency
        self.dataset = []

    def add_trajectories(self, episode):
        self.dataset.append(episode)
        self.dataset = self.dataset[-self.dataset_size :]

    def __iter__(self):
        if self.on_policy:
            episodes = self.dataset
            self.dataset = []
        else:
            # in the off-policy case we do not care about the counter
            episodes = random.choices(self.dataset, k=self.update_frequency)
        return iter(episodes)

    def __len__(self):
        return len(self.dataset)


def postprocess(episodes, tokenizer, format, mask_gen):
    trajectories = []
    rewards = []
    for episode in episodes:
        rewards.append(torch.Tensor(episode["rewards"]).squeeze(0))
        # if "log_probs" in episode.keys():
        #     log_probs = episode["log_probs"]
        if format == "chat":
            trajectories.append([tr for tr in episode["trajectory"][-1]][0])

    prompts = [tokenizer.apply_chat_template(tr, tokenize=True, add_generation_prompt=True) for tr in trajectories]

    out = mask_gen(prompts)
    attention_mask = (out["input_ids"] != tokenizer.pad_token_id).long()
    masks = (out["labels"] != -100).long()
    return {"input_ids": out["input_ids"], "attention_mask": attention_mask, "masks": masks, "rewards": rewards}
