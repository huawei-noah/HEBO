import time
from typing import List, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from trl.core import PPODecorators
from trl.core import convert_to_scalar
from trl.core import entropy_from_logits
from trl.core import flatten_dict
from trl.core import masked_mean
from trl.core import masked_var
from trl.core import stack_dicts
from trl.core import stats_to_np

from agent.models.llm import LLMBackend
from agent.train.ppo import PPOTrainer


class ReinforceTrainer(PPOTrainer):
    def __init__(self, config: DictConfig, llm: LLMBackend, output_dir: str, logger):
        super().__init__(config, llm, output_dir, logger)

    @PPODecorators.empty_cuda_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
    ):
        bs = self.config.batch_size

        queries, responses, scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, response_masks
        )
        scores = torch.tensor(scores, device=self.accelerator.device)
        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            if self.config.use_score_norm:
                scores = (scores - self.running.mean) / self.running.std
            else:
                scores /= self.running.std

        if self.config.score_clip is not None:
            # Score clipping
            scores_dtype = scores.dtype
            scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)

        timing = dict()
        t0 = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )

        model_inputs_names = list(model_inputs.keys())

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "scores": scores,
            "response_masks": response_masks,
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "returns": batch_dict["scores"][mini_batch_inds],
                        "response_masks": [batch_dict["response_masks"][i] for i in mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, masks = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                            response_masks=mini_batch_dict["response_masks"],
                        )
                        train_stats = self.train_minibatch(
                            None,
                            None,
                            logprobs,
                            logits,
                            vpreds,
                            masks,
                            None,
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        stats = stack_dicts(all_stats)

        stats["loss/policy"] = stats["loss/policy"].mean().detach().cpu()
        stats["loss/value"] = stats["loss/value"].mean().detach().cpu()
        stats["loss/total"] = stats["loss/total"].mean().detach().cpu()
        stats["policy/entropy"] = stats["policy/entropy"].mean().detach().cpu()
        stats["returns/mean"] = stats["returns/mean"].mean().detach().cpu()
        stats["returns/var"] = stats["returns/var"].mean().detach().cpu()
        stats["val/vpred"] = stats["val/vpred"].mean().detach().cpu()
        stats["val/error"] = stats["val/error"].mean().detach().cpu()

        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        end_of_first_prompt = mask.argmax(-1) - 1
        vf_loss = 0.5 * ((vpreds[:, end_of_first_prompt] - returns[:, None]) ** 2).mean()

        pg_losses = -(returns[:, None] - vpreds[:, end_of_first_prompt]) * logprobs
        pg_loss = masked_mean(pg_losses, mask)

        loss = pg_loss + self.config.vf_coef * vf_loss

        entropy = masked_mean(entropy_from_logits(logits), mask)

        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
            ),
        )
        return pg_loss, self.config.vf_coef * vf_loss, flatten_dict(stats)
