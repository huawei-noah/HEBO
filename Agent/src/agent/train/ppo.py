import math
import time
import warnings
from typing import List, Optional

import numpy
import numpy as np
import torch
from accelerate.utils import extract_model_from_parallel
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from trl import PPOConfig
from trl import PPOTrainer as PPOTrainerTRL
from trl import PreTrainedModelWrapper
from trl.core import WANDB_PADDING
from trl.core import PPODecorators
from trl.core import clip_by_value
from trl.core import convert_to_scalar
from trl.core import entropy_from_logits
from trl.core import flatten_dict
from trl.core import logprobs_from_logits
from trl.core import masked_mean
from trl.core import masked_var
from trl.core import stack_dicts
from trl.core import stats_to_np

import agent.train.models
from agent.models.llm import LLMBackend


class PPOTrainer(PPOTrainerTRL):
    def __init__(self, config: DictConfig, llm: LLMBackend, output_dir: str, logger):
        env_step_wise_gae = config.pop("env_step_wise_gae", True)
        kl_penalty = config.pop("kl_penalty", "kl")
        use_gradient_checkpointing = config.pop("use_gradient_checkpointing", True)
        self.constrained_likelihood = config.pop("constrained_likelihood")
        warm_up_epochs = config.pop("warm_up_epochs")
        self.entropy_coef = config.pop("entropy_coef")

        model = llm.model
        tokenizer = llm.tokenizer
        if use_gradient_checkpointing:
            model.pretrained_model.enable_input_require_grads()
            model.gradient_checkpointing_enable()

        trainable_params, all_param = model.pretrained_model.get_nb_trainable_parameters()
        logger.log_metrics(
            {
                "trainable_params": trainable_params,
                "all_param": all_param,
                "trainable": 100 * trainable_params / all_param,
            }
        )

        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
        )
        previous_optimizer = config.pop("last_optimizer_path", None)
        if previous_optimizer is not None:
            previous_optimizer = torch.load(previous_optimizer, map_location="cpu")
            optimizer.state = previous_optimizer["state"]

        # if detached head, we don't have to use a lr scheduler, only disable policy loss
        self.warm_up_epochs = warm_up_epochs
        if isinstance(model, agent.train.models.AutoModelForCausalLMWithDetachedValueHead):
            self.warm_up_scheduler = False
            lr_scheduler = None
            logger.log_metrics({"warm_up_disable_pol_loss": warm_up_epochs})
        else:
            self.warm_up_scheduler = True
            lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.0 if epoch < warm_up_epochs else 1)
            logger.log_metrics({"warm_up_scheduler": warm_up_epochs})

        logger.save_metrics("all")

        ppo_config = PPOConfig(
            batch_size=config.buffer_size,
            learning_rate=config.learning_rate,
            mini_batch_size=config.mini_batch_size,
            ppo_epochs=config.optim_epochs,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            cliprange=config.cliprange,
            tracker_project_name="tensorboard",
            remove_unused_columns=False,
            optimize_device_cache=True,
            log_with="tensorboard",
            project_kwargs=dict(logging_dir=output_dir),
        )
        ppo_config.env_step_wise_gae = env_step_wise_gae
        ppo_config.kl_penalty = kl_penalty
        self.max_training_epochs = config.training_epochs
        self.training_epochs_counter = 0
        self.output_dir = output_dir
        self.best_perf = float("-inf")
        self.logger = logger

        super().__init__(ppo_config, model, None, tokenizer, None, optimizer, None, None, lr_scheduler=lr_scheduler)

    def done(self):
        return self.training_epochs_counter >= self.max_training_epochs

    def train(self, queries, responses, response_masks, choices_masks, scores, failed_rewards, elapsed_collection=0):
        # merge and convert tensor from all workers
        queries = [torch.tensor(sublist) for q in queries for sublist in q]
        responses = [torch.tensor(sublist) for r in responses for sublist in r]
        response_masks = [torch.tensor(sublist) for rm in response_masks for sublist in rm]
        scores = [torch.tensor(sublist, dtype=torch.float) for s in scores for sublist in s]
        failed_rewards = [torch.tensor(sublist, dtype=torch.float) for s in failed_rewards for sublist in s]

        if self.constrained_likelihood:
            choices_masks = [sublist for cm in choices_masks for sublist in cm]
        else:
            choices_masks = None

        self.accelerator.wait_for_everyone()
        start = time.time()
        self.model.train()
        train_stats = self.step(queries, responses, scores, failed_rewards, response_masks, choices_masks)
        elapsed_update = time.time() - start

        # logging
        train_stats["time/elapsed_collection"] = elapsed_collection
        train_stats["time/elapsed_update"] = elapsed_update
        first_layer = extract_model_from_parallel(self.model).pretrained_model.base_model.model.model.layers[0]
        train_stats["one_weight"] = first_layer.self_attn.q_proj.lora_A.default.weight[0, 0].item()

        self.log_stats(train_stats, {}, scores)

        if self.accelerator.is_main_process:
            self.logger.log_metrics(
                {k: v for k, v in train_stats.items() if type(v) != numpy.ndarray or v.shape[-1] == 1}
            )
            self.logger.save_metrics("all")

            if train_stats["ppo/mean_scores"] >= self.best_perf:
                self.best_perf = train_stats["ppo/mean_scores"]
                self.save_pretrained(f"{self.output_dir}/best.pt")

            if self.training_epochs_counter % 10 == 0 and self.training_epochs_counter > 0:
                self.save_pretrained(f"{self.output_dir}/snapshot.pt")

        self.training_epochs_counter += 1
        self.warm_up_epochs -= 1

    # fixed a problem with response_masks
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
        response_masks: Optional[torch.Tensor] = None,
        choices_masks=None,
    ):
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            logits, _, values = model(**input_kwargs)

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            logits_for_logprobs = logits[:, :-1, :]
            if choices_masks is not None:
                cm = choices_masks[i * fbs : (i + 1) * fbs][:, 1:].to(self.current_device)
                logits_for_logprobs[cm] = float("-inf")

            logprobs = logprobs_from_logits(logits_for_logprobs, input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = len(query_batch[j]) - 1
                    if attention_mask[j, 0] == 0:  # offset left padding
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])
                    if response_masks is not None:
                        # fixed TRL here
                        response_masks_batch[j] = torch.cat(
                            (torch.zeros_like(input_ids[j, : -response_batch[j].shape[0]]), response_masks_batch[j])
                        )[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    # fix deepspeed by calling accelerator methods instead of dist ones
    def gather_stats(self, stats):
        """Gather stats from all processes. Useful in the context of distributed training.

        Args:
            stats (dict[str, Any]):
            a dictionary of stats to be gathered. The stats should contain torch tensors.

        Returns:
            `dict[str, Any]`: A dictionary of stats with the tensors gathered.
        """
        # Wait for all processes to finish
        self.accelerator.wait_for_everyone()

        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                v = self.accelerator.reduce(v.to(self.accelerator.device), reduction="sum")
                v /= self.accelerator.num_processes
            stats[k] = v
        return stats

    def _kl_penalty(self, logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor) -> torch.FloatTensor:
        if self.config.kl_penalty == "none":
            return torch.zeros_like(logprob)

        return super()._kl_penalty(logprob, ref_logprob)

    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
        mask_values: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
        choices_masks: torch.FloatTensor,
    ):
        self.model.train()
        loss_p, loss_v, train_stats = self.loss(
            old_logprobs, values, logits, vpreds, logprobs, mask, mask_values, advantages, returns, choices_masks
        )
        if not self.warm_up_scheduler and self.warm_up_epochs > 0:
            loss = loss_p * 0.0 + loss_v
        else:
            loss = loss_p + loss_v
        self.accelerator.backward(loss)
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model_params, self.config.max_grad_norm)
        self.optimizer.step()
        # we call optimizer.zero_grad() every time and let `accelerator` handle accumulation
        # see https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#the-finished-code
        self.optimizer.zero_grad()
        return train_stats

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
        mask_values: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
        choices_masks: torch.FloatTensor,
    ):
        vpredclipped = clip_by_value(
            vpreds,
            values - self.config.cliprange_value,
            values + self.config.cliprange_value,
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask_values)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask_values)
        ratio = torch.exp(logprobs - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        if self.constrained_likelihood:
            ent_mask = ~choices_masks.to(self.current_device)[:, 1:, :]
            ent_mask[~(mask.bool())] = False
            log_pd = logits.log_softmax(-1)
            entropy = -((log_pd.exp() * log_pd)[ent_mask]).sum(-1) / mask.sum()
        else:
            entropy = masked_mean(entropy_from_logits(logits), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss + self.entropy_coef * -entropy

        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.config.ratio_threshold:
            warnings.warn(
                f"The average ratio of batch ({avg_ratio:.2f}) exceeds threshold {self.config.ratio_threshold:.2f}. "
                f"Skipping batch."
            )
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            loss = loss * 0.0

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        return_mean, return_var = masked_mean(returns, mask_values), masked_var(returns, mask_values)
        value_mean, value_var = masked_mean(values, mask_values), masked_var(values, mask_values)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )
        return pg_loss, self.config.vf_coef * vf_loss, flatten_dict(stats)

    # just add a needed argument to compute_advantage for env step-wise advantage + cmask
    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        failed_rewards: List[torch.FloatTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
        choices_masks=None,
    ):
        """Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            response_masks (List[`torch.FloatTensor`], *optional*)):
                List of tensors containing masks of the response tokens.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        bs = self.config.batch_size

        queries, responses, scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, response_masks
        )
        scores = torch.tensor(scores)
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

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = scores.mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

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

        # also pad choices_masks
        if choices_masks is not None:
            assert self.tokenizer.padding_side == "left"
            cm = torch.zeros((len(queries), model_inputs.input_ids.shape[1], 32000), dtype=torch.bool)
            for i, masks in enumerate(choices_masks):
                pad_idx = model_inputs["attention_mask"][i].argmax()
                for j, mask in enumerate(masks):
                    if mask is not None:
                        cm[i, j + pad_idx] = True
                        cm[i, j + pad_idx][mask] = False
            choices_masks = cm

        model_inputs_names = list(model_inputs.keys())

        full_kl_penalty = self.config.kl_penalty == "full"

        with torch.no_grad():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
                choices_masks=choices_masks,
            )
            nan_detect = all_logprobs.isnan()
            if nan_detect.any() or all_logprobs.isinf().any():
                print(f"Nan already in pure inference in {nan_detect.any(-1).float().mean().item()*100.}%.")
                exit(1)

            if self.config.kl_penalty != "none":
                # for when the model is a peft model
                if self.is_peft_model and hasattr(
                    self.accelerator.unwrap_model(self.model).pretrained_model,
                    "disable_adapter",
                ):
                    with self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter():
                        ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                            self.model,
                            queries,
                            responses,
                            model_inputs,
                            return_logits=full_kl_penalty,
                            choices_masks=choices_masks,
                        )
                elif self.is_peft_model and not hasattr(self.model.pretrained_model, "disable_adapter"):
                    raise ValueError(
                        "You are using a `peft` version that does not support `disable_adapter`. "
                        "Please update your `peft` version to the latest version."
                    )

                else:
                    ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                        self.ref_model,
                        queries,
                        responses,
                        model_inputs,
                        return_logits=full_kl_penalty,
                        choices_masks=choices_masks,
                    )
            else:
                ref_logprobs = torch.ones_like(all_logprobs)

        timing["time/ppo/forward_pass"] = time.time() - t

        mask_values = None
        with torch.no_grad():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward = self.compute_rewards(scores, active_full_logprobs, ref_full_logprobs, masks)
            else:
                rewards, non_score_reward = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns, mask_values = self.compute_advantages(values, rewards, masks, non_score_reward)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "mask_values": mask_values if mask_values is not None else masks,
            "advantages": advantages,
            "returns": returns,
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
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        "mask_values": batch_dict["mask_values"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                            choices_masks=choices_masks[mini_batch_inds] if choices_masks is not None else None,
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["mask_values"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                            choices_masks=choices_masks[mini_batch_inds] if choices_masks is not None else None,
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
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats

    def compute_advantages(
        self: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        mask: torch.FloatTensor,
        non_score_reward: torch.FloatTensor,
    ):
        if not self.config.env_step_wise_gae:
            values, advantages, returns = super().compute_advantages(values, rewards, mask)
            return values, advantages, returns, None

        # env-step wise GAE
        score_only_rewards = rewards - non_score_reward
        score_only_rewards = score_only_rewards * mask

        # TODO: optimise me, maybe with torch.unique_consecutive
        score_only_advantages_factored = []
        advantages = torch.zeros_like(values)
        power_counter = torch.zeros_like(mask)
        mask_values = torch.zeros_like(mask)
        returns = torch.zeros_like(values)
        values_unroll_mapping = torch.zeros_like(mask, dtype=torch.int)
        values_score_only = []
        score_only_rewards_factored = []
        for i in range(mask.shape[0]):
            counter = 1
            values_score_only_i = []
            non_scores_rewards_ij = []
            score_only_rewards_factored_i = []
            score_only_rewards_factored_ij = []
            for j in reversed(range(mask.shape[1])):
                if mask[i, j] != 0:
                    power_counter[i, j] = counter
                    score_only_rewards_factored_ij.append(score_only_rewards[i, j])
                    non_scores_rewards_ij.append(non_score_reward[i, j])
                    if j > 0 and mask[i, j] == 1 and mask[i, j - 1] == 0:
                        values_unroll_mapping[i, j - 1] = counter
                        counter += 1
                        score_only_rewards_factored_i.append(torch.stack(score_only_rewards_factored_ij).max())
                        if self.config.kl_penalty != "none":
                            score_only_rewards_factored_i[-1] += torch.stack(non_scores_rewards_ij).mean()
                        score_only_rewards_factored_ij = []
                        non_scores_rewards_ij = []
                        mask_values[i, j - 1] = 1
                        values_score_only_i.append(values[i, j - 1])

            values_score_only.append(torch.stack(values_score_only_i).flip(-1))
            score_only_rewards_factored.append(torch.stack(score_only_rewards_factored_i).flip(-1))
            lastgaelam = 0
            advantages_reversed_i = []
            for t in reversed(range(values_score_only[-1].shape[0])):
                nextvalues = values_score_only[-1][t + 1] if t < values_score_only[-1].shape[0] - 1 else 0.0
                delta = score_only_rewards_factored[-1][t] + self.config.gamma * nextvalues - values_score_only[-1][t]
                lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
                advantages_reversed_i.append(lastgaelam)
            score_only_advantages_factored.append(torch.stack(advantages_reversed_i[::-1]))

            unroll_mapping_i = counter - 1 - power_counter[i]
            values_unroll_mapping_i = counter - 1 - values_unroll_mapping[i]
            soaf_pad = torch.cat((score_only_advantages_factored[-1], torch.zeros_like(rewards[0, 0:1])), 0)

            advantages[i] = soaf_pad[unroll_mapping_i]
            returns[i] = soaf_pad[values_unroll_mapping_i] + values[i] * mask_values[i]

        score_only_advantages_factored = torch.cat(score_only_advantages_factored)
        mean, var = score_only_advantages_factored.mean(), score_only_advantages_factored.var()
        advantages = (advantages - mean) * torch.rsqrt(var + 1e-8)

        advantages = advantages.detach()
        return values, advantages, returns, mask_values
