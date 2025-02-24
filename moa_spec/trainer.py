# MIT License
#
# Copyright (c) 2024, Huawei Technologies Co., Ltd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gc
import logging
import os
import time

import hydra
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from moa_spec.models.train.moa_spec import MOASpecLlamaForCausalLM, MOASpecQwen2ForCausalLM
from moa_spec.utils import DataCollatorWithPadding, convert_list_of_dicts_to_dict_of_lists, \
    zero_peak_constant_scheduler

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, tokenizer, training_dataset, validation_dataset, rank, world_size,
                 batch_size, mini_batch_size, epochs, learning_rate, weight_decay, warmup_steps,
                 zero_peak_constant, max_grad_norm=None):

        self.rank = rank
        self.world_size = world_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm
        self.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.forward_use_cache = (
                isinstance(model, MOASpecLlamaForCausalLM) or
                isinstance(model, MOASpecQwen2ForCausalLM)
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_dataloader = DataLoader(
            training_dataset,
            batch_size=mini_batch_size,
            shuffle=True,
            collate_fn=data_collator,
            pin_memory=True
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=mini_batch_size,
            shuffle=True,
            collate_fn=data_collator,
            pin_memory=True
        )

        self.trainable_parameters = [param for param in model.parameters() if param.requires_grad]
        number_of_trainable_params = sum([param.numel() for param in self.trainable_parameters])

        if ("ACCELERATE_MIXED_PRECISION" in os.environ and
                os.environ["ACCELERATE_MIXED_PRECISION"] in ["fp16", "bf16"]):
            for p in self.trainable_parameters:
                p.data = p.data.float()

        gradient_accumulation_steps = batch_size // (world_size * mini_batch_size)
        self.trainable_dtype = self.trainable_parameters[0].dtype

        if rank == 0:
            logger.info(f"parameter_precision: {next(model.parameters()).dtype}")
            logger.info(f"trainable_parameter_precision: {self.trainable_dtype}")
            logger.info(f"full_trainable_params: {number_of_trainable_params}")

            logger.info(f"total batch size: {batch_size}")
            logger.info(f"mini batch size: {mini_batch_size}")
            logger.info(f"computed gradient accumulation steps: {gradient_accumulation_steps}")

        assert gradient_accumulation_steps * (world_size * mini_batch_size) == batch_size

        # Initialize the optimizer
        optimizer = AdamW(
            self.trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas = (0.9, 0.95)
        )

        # Initialize the learning rate scheduler
        lr_scheduler = LambdaLR(optimizer,
            lr_lambda=zero_peak_constant_scheduler(
                zero_warmup_steps=warmup_steps,
                decay_steps=len(training_dataset) // batch_size,
                peak_lr=20 * learning_rate,
                constant_lr=learning_rate,
            ) if zero_peak_constant else
                zero_peak_constant_scheduler(
                    zero_warmup_steps=warmup_steps,
                    decay_steps=warmup_steps,
                    peak_lr=learning_rate,
                    constant_lr=learning_rate,
            )
        )

        # Initialize the accelerator
        mixed_precision = (
            os.environ["ACCELERATE_MIXED_PRECISION"] if "ACCELERATE_MIXED_PRECISION" in os.environ else "no"
        )

        dataloader_config = DataLoaderConfiguration(non_blocking=True)
        self.accelerator = Accelerator(
            log_with="tensorboard",
            gradient_accumulation_steps=gradient_accumulation_steps,
            project_config=ProjectConfiguration(logging_dir=self.output_dir),
            mixed_precision=mixed_precision,
            dataloader_config=dataloader_config
        )
        self.is_distributed = self.accelerator.distributed_type in ["MULTI_GPU", "DEEPSPEED"]
        assert self.accelerator.num_processes == self.world_size
        len_training_dataloader = len(training_dataloader)

        # don't pass lr_scheduler as it'd need more sync between workers
        self.lr_scheduler = lr_scheduler
        self.model, self.training_dataloader, self.validation_dataloader, self.optimizer = (
            self.accelerator.prepare(
                model,
                training_dataloader,
                validation_dataloader,
                optimizer,
            )
        )

        if self.rank == 0:
            self.accelerator.init_trackers("tensorboard")

            logger.info(f"Mixed precision enabled: {mixed_precision}")
            logger.info(f"Dataloader reduces from {len_training_dataloader} to {len(self.training_dataloader)} elements "
                        f"after splitting across {self.world_size} GPU(s)\n"
                        f"{self.accelerator.split_batches=} "
                        f"{self.accelerator.distributed_type=} "
                        f"{self.accelerator.use_distributed=} "
                        f"{self.accelerator.mixed_precision=}")

        if self.accelerator.distributed_type == "DEEPSPEED":
            # should not store a references to parameters when using deepspeed
            del self.trainable_parameters
            logger.warning(f"Gradient clipping is handled by deepspeed, {self.max_grad_norm} is ignored.")

        gc.collect()
        torch.cuda.empty_cache()

    def train(self):

        all_stats = []
        step = 0
        training_step = 0
        epoch = 0
        train = False
        new_epoch = False
        training_iterator = iter(self.training_dataloader)
        validation_iterator = iter(self.validation_dataloader)
        start_time = time.time()

        while epoch < self.epochs:
            # decide training or eval
            if all_stats == [] and step % 20 == 0:
                train = False

            if train:
                self.model.train()
                current_iterator = training_iterator
            else:
                self.model.eval()
                current_iterator = validation_iterator

            try:
                input_ids, attention_mask, response_mask = next(current_iterator)
            except StopIteration:
                if train:
                    current_iterator= training_iterator = iter(self.training_dataloader)
                    epoch += 1
                    new_epoch = True
                else:
                    current_iterator = validation_iterator = iter(self.validation_dataloader)
                input_ids, attention_mask, response_mask = next(current_iterator)

            # must be called outside autocast or it's too slow
            with torch.no_grad():
                # use a list to clean memory inside self.model forward call
                base_model_output = [self.get_internal_module().model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=self.forward_use_cache
                )]

            with (
                self.accelerator.accumulate(self.model),
                torch.set_grad_enabled(train),
            ):
                loss, stats = self.model(
                    base_model_output=base_model_output,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    response_mask=response_mask,
                    epoch=epoch,
                    max_epochs=self.epochs
                )

                if train:
                    self.accelerator.backward(loss)
                    if self.max_grad_norm is not None and self.accelerator.distributed_type != "DEEPSPEED":
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.trainable_parameters, self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            all_stats.append(stats)

            # if len(all_stats) * self.mini_batch_size * self.world_size == self.batch_size
            if self.accelerator.sync_gradients:
                all_stats = convert_list_of_dicts_to_dict_of_lists(all_stats)

                stats = {}
                for k, v in all_stats.items():
                    stats[f"{'train' if train else 'valid'}/{k}"] = torch.mean(torch.stack(v))

                if self.is_distributed:
                    self.accelerator.wait_for_everyone()

                    for k, v in stats.items():
                        v = self.accelerator.reduce(v.to(self.accelerator.device), reduction="sum")
                        v /= self.accelerator.num_processes
                        stats[k] = v.item()
                else:
                    stats = {k:v.item() for k,v in stats.items()}

                if train:
                    stats["learning_rate"] = self.optimizer.param_groups[0]["lr"]
                stats["one_weight"] = self.get_one_weight()
                stats["iteration_time"] = time.time() - start_time

                if self.accelerator.scaler is not None:
                    stats["scaler"] = self.accelerator.scaler.get_scale()
                elif hasattr(self.accelerator, "deepspeed_engine_wrapped"):
                    stats["scaler"] = self.accelerator.deepspeed_engine_wrapped.engine.optimizer.loss_scaler.loss_scale

                stats["training_step"] = training_step
                self.accelerator.log(stats, step=training_step)

                if train:
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                # update the current step
                step += 1
                if train:
                    training_step += 1

                if self.accelerator.is_main_process:
                    logger.info(stats)

                    if train and new_epoch:
                        self.save_model(f"{self.output_dir}/snapshot.{training_step}.pt")
                        new_epoch = False

                all_stats = []
                train = True
                start_time = time.time()

    def save_model(self, output_path):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            output_path,
            save_function=self.accelerator.save,
        )
        logger.info(f"Model saved to {output_path}")

    def get_one_weight(self):
        model = self.get_internal_module()
        if hasattr(model, "self_attention"):
            last_layer = model.self_attention
        elif isinstance(model.spec_model, torch.nn.ModuleList):
            last_layer = model.spec_model[0]
        elif hasattr(model.spec_model, "model"):
            last_layer = model.spec_model.model.layers[0]
        else:
            last_layer = model.spec_model

        return last_layer.self_attn.v_proj.weight[0, 0].item()

    def get_internal_module(self):
        return self.model.module if self.is_distributed else self.model
