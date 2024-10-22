# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from packaging import version
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import time
import functools
import os
import gc
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from huggingface_hub.utils._deprecation import _deprecate_arguments
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    is_wandb_available
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import (
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_accelerate_available,
    is_torch_neuroncore_available,
    is_torch_xla_available,
    XLA_FSDPV2_MIN_VERSION,
    logging,
)

from transformers.utils import is_peft_available
from transformers.trainer import _is_peft_model
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from accelerate.utils import DistributedType

from .sparse_config import SparseConfig
from trl.models import PreTrainedModelWrapper, create_reference_model
from .mapo_config import MAPOConfig
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    RunningMoments,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False

logger = logging.get_logger(__name__)


class SparseTrainer(Trainer):
    r"""
    Initialize TDPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        args (`TDPOConfig`):
            The TDPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "tdpo", "sparse"]

    @_deprecate_arguments(
        version="1.0.0",
        deprecated_args=[
            "beta",
            "alpha",
            "version",
            "label_smoothing",
            "loss_type",
            "label_pad_token_id",
            "padding_value",
            "truncation_mode",
            "max_length",
            "max_prompt_length",
            "max_target_length",
            "is_encoder_decoder",
            "disable_dropout",
            "generate_during_eval",
            "precompute_ref_log_probs",
            "dataset_num_proc",
            "model_init_kwargs",
            "ref_model_init_kwargs",
            "model_adapter_name",
            "ref_adapter_name",
            "reference_free",
            "force_use_ref_model",
        ],
        custom_message="Deprecated positional argument(s) used in TDPOTrainer, please use the TDPOConfig to set these arguments instead.",
    )
    def __init__(
            self,
            model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            beta: float = 0.1,
            alpha: float = 1.0,
            version: int = 1,
            label_smoothing: float = 0,
            loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair", "bco_pair", "robust"] = "sigmoid",
            args: Optional[SparseConfig] = None,
            data_collator: Optional[DataCollator] = None,
            label_pad_token_id: int = -100,
            padding_value: Optional[int] = None,
            truncation_mode: str = "keep_end",
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            max_length: Optional[int] = None,
            max_prompt_length: Optional[int] = None,
            max_target_length: Optional[int] = None,
            peft_config: Optional[Dict] = None,
            is_encoder_decoder: Optional[bool] = None,
            disable_dropout: bool = True,
            generate_during_eval: bool = False,
            compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
            precompute_ref_log_probs: bool = False,
            dataset_num_proc: Optional[int] = None,
            model_init_kwargs: Optional[Dict] = None,
            ref_model_init_kwargs: Optional[Dict] = None,
            model_adapter_name: Optional[str] = None,
            ref_adapter_name: Optional[str] = None,
            reference_free: bool = False,
            force_use_ref_model: bool = False,
    ):
        if model_init_kwargs is not None:
            warnings.warn(
                "You passed `model_init_kwargs` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.model_init_kwargs = model_init_kwargs

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_init_kwargs to the TDPOTrainer/TDPOConfig, but your model is already instantiated."
            )
        else:
            model_init_kwargs = args.model_init_kwargs
            model_init_kwargs["torch_dtype"] = (
                model_init_kwargs["torch_dtype"]
                if model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, model_init_kwargs["torch_dtype"])
            )

        if ref_model_init_kwargs is not None:
            warnings.warn(
                "You passed `ref_model_init_kwargs` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.ref_model_init_kwargs = ref_model_init_kwargs

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_init_kwargs to the TDPOTrainer/TDPOConfig, but your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            ref_model_init_kwargs["torch_dtype"] = (
                ref_model_init_kwargs["torch_dtype"]
                if ref_model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, ref_model_init_kwargs["torch_dtype"])
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the TDPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the TDPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if force_use_ref_model:
            warnings.warn(
                "You passed `force_use_ref_model` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.force_use_ref_model = force_use_ref_model

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None and not args.force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in TDPOTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval:
            warnings.warn(
                "You passed `generate_during_eval` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.generate_during_eval = generate_during_eval
        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if is_encoder_decoder is not None:
            warnings.warn(
                "You passed `is_encoder_decoder` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.is_encoder_decoder = is_encoder_decoder
        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError(
                "When no model is provided, you need to pass the parameter is_encoder_decoder to the TDPOTrainer/TDPOConfig."
            )
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        if model_adapter_name is not None:
            warnings.warn(
                "You passed `model_adapter_name` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.model_adapter_name = model_adapter_name
        self.model_adapter_name = args.model_adapter_name

        if ref_adapter_name is not None:
            warnings.warn(
                "You passed `ref_adapter_name` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.ref_adapter_name = ref_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if reference_free:
            warnings.warn(
                "You passed `reference_free` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.reference_free = reference_free
        self.reference_free = args.reference_free

        if precompute_ref_log_probs:
            warnings.warn(
                "You passed `precompute_ref_log_probs` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.precompute_ref_log_probs = precompute_ref_log_probs

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        # Learnable Masks
        # if mask:
        #     mask = mask #.half()

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")

        if max_length is not None:
            warnings.warn(
                "You passed `max_length` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.max_length = max_length
        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set in the TDPOConfig's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_length = 512

        if max_prompt_length is not None:
            warnings.warn(
                "You passed `max_prompt_length` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.max_prompt_length = max_prompt_length
        if args.max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the TDPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_prompt_length = 128

        if max_target_length is not None:
            warnings.warn(
                "You passed `max_target_length` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.max_target_length = max_target_length
        if args.max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the TDPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_target_length = 128

        if label_pad_token_id != -100:
            warnings.warn(
                "You passed `label_pad_token_id` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.label_pad_token_id = label_pad_token_id
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if not disable_dropout:
            warnings.warn(
                "You passed `disable_dropout` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.disable_dropout = disable_dropout
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = args.max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        if padding_value is not None:
            warnings.warn(
                "You passed `padding_value` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.padding_value = padding_value
        self.padding_value = args.padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = args.max_prompt_length
        if truncation_mode != "keep_end":
            warnings.warn(
                "You passed `truncation_mode` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.truncation_mode = truncation_mode
        self.truncation_mode = args.truncation_mode
        self.max_target_length = args.max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = args.precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        if loss_type != "sigmoid":
            warnings.warn(
                "You passed `loss_type` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.loss_type = loss_type
        if label_smoothing != 0:
            warnings.warn(
                "You passed `label_smoothing` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.label_smoothing = label_smoothing
        if args.loss_type in ["hinge", "ipo", "kto_pair", "bco_pair"] and args.label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        if beta != 0.1:
            warnings.warn(
                "You passed `beta` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.beta = beta

 
        self.beta = args.beta

        # args.distributed_state.distributed_type = DistributedType.DEEPSPEED

        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self.l0_norm_param = args.l0_norm_param
        self.l1_norm_param_u = args.l1_norm_param_u
        self.l1_norm_param_d = args.l1_norm_param_d

        self.mask_config = {
            'mask_model': args.mask_model,
            'mask_on_reward': args.mask_on_reward,
            'mask_on_kl': args.mask_on_kl,
            'is_rw_kl_independent': args.rw_kl_independent
        }

        # case when common mask -> make up for double adding (from u and d)
        if args.mask_on_reward and args.mask_on_kl and not args.rw_kl_independent:
            self.l1_norm_param_u /= 2.0
            self.l1_norm_param_d /= 2.0

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        if dataset_num_proc is not None:
            warnings.warn(
                "You passed `dataset_num_proc` to the TDPOTrainer, the value you passed will override the one in the `TDPOConfig`."
            )
            args.dataset_num_proc = dataset_num_proc
        self.dataset_num_proc = args.dataset_num_proc

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset
            train_dataset = train_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
            if args.sync_ref_model:
                raise ValueError(
                    "You currently cannot use `ref_model=None` with TR-DPO method. Please provide `ref_model`."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            if precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with TR-DPO method. Please set `precompute_ref_log_probs=False`."
                )

            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))
        if self.loss_type == "bco_pair":
            self.running = RunningMoments(self.accelerator)

        ##

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            eval_dataset = eval_dataset.add_column(name="reference_chosen_logps", column=all_reference_chosen_logps)
            eval_dataset = eval_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            # Save calculated reference_chosen_logps and reference_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids):]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids):]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            # Last prompt token might get merged by tokenizer and
            # it should not be included for generation if that happens
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Make sure prompts only have one different token at most an
            # and length only differs by 1 at most
            num_diff_tokens = sum(
                [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
            )
            num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )

            # add BOS token to head of prompt. Avoid adding if it's already there
            bos_token_id = self.tokenizer.bos_token_id

            if bos_token_id is not None:
                if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
                    prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
                    prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
                if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
                    chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
                    chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
                if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
                    rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
                    rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer. Avoid adding if it's already there
            eos_token_id = self.tokenizer.eos_token_id
            if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
                chosen_tokens["input_ids"].append(eos_token_id)
                chosen_tokens["attention_mask"].append(1)
            if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
                rejected_tokens["input_ids"].append(eos_token_id)
                rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length:]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                                                                                             self.label_pad_token_id
                                                                                         ] * len(
                chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                                                                                                 self.label_pad_token_id
                                                                                             ] * len(
                rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["rejected_labels"])
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"])
                )

        return batch

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
                self.model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                        _, _, _
                    ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    _, _, _, _
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps

    @staticmethod
    def concatenated_inputs(
            batch: Dict[str, Union[List, torch.LongTensor]],
            is_encoder_decoder: bool = False,
            label_pad_token_id: int = -100,
            padding_value: int = 0,
            device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    @staticmethod
    def get_batch_logits(
            logits: torch.FloatTensor,
            labels: torch.LongTensor,
            label_pad_token_id: int = -100,
            is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        return logits, loss_mask, labels

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]],
        ref_chosen_states: torch.LongTensor = None,
        ref_rejected_states: torch.LongTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """

        run_mask = ref_chosen_states is not None and ref_rejected_states is not None
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
                "return_dict": True,
            }
            if self.is_encoder_decoder
            else {}
        )
        if run_mask:
            model_kwargs['reference_states'] = [torch.concat([x,y],dim=0) for x,y in zip(ref_chosen_states,ref_rejected_states)]
            if not self.is_encoder_decoder:
                labels = concatenated_batch["concatenated_labels"][:, 1:].clone()
            loss_mask = labels != self.label_pad_token_id
            model_kwargs['loss_mask'] = loss_mask

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            output_hidden_states=True,
            **model_kwargs,
        )

        all_logits, hidden_states,  = outputs.logits, outputs.hidden_states, 

        logits, loss_mask, labels = self.get_batch_logits(
            all_logits,
            concatenated_batch["concatenated_labels"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logits = logits[:len_chosen]
        rejected_logits = logits[len_chosen:]

        chosen_loss_mask = loss_mask[:len_chosen]
        rejected_loss_mask = loss_mask[len_chosen:]

        chosen_rw_mask,rejected_rw_mask = None, None
        chosen_kl_mask, rejected_kl_mask = None, None
        if run_mask:
            rw_mask, kl_mask = outputs.reward_mask, outputs.kl_mask
            chosen_rw_mask = rw_mask[:len_chosen]
            rejected_rw_mask = rw_mask[len_chosen:]
            chosen_kl_mask = kl_mask[:len_chosen]
            rejected_kl_mask = kl_mask[len_chosen:]

        chosen_labels = labels[:len_chosen]
        rejected_labels = labels[len_chosen:]
        
        chosen_hidden_states = [x[:len_chosen] for x in hidden_states]
        rejected_hidden_states = [x[len_chosen:] for x in hidden_states]

        return (chosen_logits, rejected_logits,
                chosen_rw_mask, rejected_rw_mask,
                chosen_kl_mask, rejected_kl_mask,
                chosen_loss_mask, rejected_loss_mask,
                chosen_labels, rejected_labels,
                chosen_hidden_states, rejected_hidden_states)

    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
        """Compute mean of tensor with a masked values."""
        if axis is not None:
            return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
        else:
            return (values * mask).sum() / mask.sum()

    def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}


        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if (
            "reference_chosen_logps" in batch
            and "reference_rejected_logps" in batch
            and self.args.rpo_alpha is not None
        ):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logits,
                            reference_rejected_logits,
                            _, _, _, _,
                            _, _, _, _,
                            ref_chosen_hidden_states,
                            ref_rejected_hidden_states
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logits,
                        reference_rejected_logits,
                        _, _, _, _,
                        _, _, _, _,
                        ref_chosen_hidden_states,
                        ref_rejected_hidden_states
                    ) = self.concatenated_forward(self.ref_model, batch)

        (
            policy_chosen_logits,
            policy_rejected_logits,
            rw_mask_chosen,rw_mask_rejected,
            kl_mask_chosen,kl_mask_rejected,
            chosen_loss_mask,
            rejected_loss_mask,
            chosen_labels,
            rejected_labels,
            pol_chosen_hidden_states, 
            pol_rejected_hidden_states
        ) = self.concatenated_forward(model, batch,
                                      ref_chosen_states=ref_chosen_hidden_states,
                                      ref_rejected_states=ref_rejected_hidden_states)

        # Start by computing mask * (log pi - log pi_r)
        policy_chosen_logps_vocab = policy_chosen_logits.log_softmax(-1)
        policy_rejected_logps_vocab = policy_rejected_logits.log_softmax(-1)
        reference_chosen_logps_vocab = reference_chosen_logits.log_softmax(-1)
        reference_rejected_logps_vocab = reference_rejected_logits.log_softmax(-1)

        policy_chosen_token_logps = torch.gather(policy_chosen_logps_vocab, dim=2, index=chosen_labels.unsqueeze(2)).squeeze(2)  # (b, seq)
        policy_rejected_token_logps = torch.gather(policy_rejected_logps_vocab, dim=2, index=rejected_labels.unsqueeze(2)).squeeze(2)  # (b, seq)
        reference_chosen_token_logps = torch.gather(reference_chosen_logps_vocab, dim=2, index=chosen_labels.unsqueeze(2)).squeeze(2)  # (b, seq)
        reference_rejected_token_logps = torch.gather(reference_rejected_logps_vocab, dim=2, index=rejected_labels.unsqueeze(2)).squeeze(2)  # (b, seq)

        chosen_reward = policy_chosen_token_logps - reference_chosen_token_logps
        rejected_reward = policy_rejected_token_logps - reference_rejected_token_logps

        if self.mask_config.get('mask_on_reward',False):
            chosen_reward = rw_mask_chosen * chosen_reward
            rejected_reward = rw_mask_rejected * rejected_reward
        
        chosen_reward = (chosen_reward * chosen_loss_mask).sum(-1)
        rejected_reward = (rejected_reward * rejected_loss_mask).sum(-1)  # (b,)
        u = chosen_reward - rejected_reward

        # Token-level KL divergences
        reference_chosen_ps = reference_chosen_logits.softmax(-1)  # (b, seq, vocab) distribution over vocab
        reference_rejected_ps = reference_rejected_logits.softmax(-1)

        chosen_token_kl = (reference_chosen_ps * (policy_chosen_logps_vocab - reference_chosen_logps_vocab)).sum(-1)  # (b, seq) sum across the vocab dist
        rejected_token_kl = (reference_rejected_ps * (policy_rejected_logps_vocab - reference_rejected_logps_vocab)).sum(-1)

        premask_chosen_token_kl = chosen_token_kl
        premask_rejected_token_kl = rejected_token_kl
        
        if self.mask_config.get('mask_on_kl',False):
            chosen_token_kl = kl_mask_chosen * chosen_token_kl
            rejected_token_kl = kl_mask_rejected * rejected_token_kl
        #
        chosen_token_kl = (chosen_token_kl * chosen_loss_mask).sum(-1)  # (b,)
        rejected_token_kl = (rejected_token_kl * rejected_loss_mask).sum(-1) # (b,)
        d = chosen_token_kl - rejected_token_kl

        #######
        # LOSS
        #######
        losses = -F.logsigmoid(self.beta * (u - d))

        norm_loss_chosen_u,norm_loss_rejected_u = 0,0
        norm_loss_chosen_d,norm_loss_rejected_d = 0,0
        
        if self.mask_config.get('mask_on_reward',False):
            norm_loss_chosen_u =  torch.linalg.vector_norm(rw_mask_chosen * chosen_loss_mask, ord=1, dim=-1)
            norm_loss_rejected_u = torch.linalg.vector_norm(rw_mask_rejected * rejected_loss_mask, ord=1, dim=-1)
            if self.l1_norm_param_u != 0:
                losses += self.l1_norm_param_u * (norm_loss_chosen_u + norm_loss_rejected_u)
        
        if self.mask_config.get('mask_on_kl',False):
            norm_loss_chosen_d = torch.linalg.vector_norm(kl_mask_chosen * chosen_loss_mask, ord=1, dim=-1)
            norm_loss_rejected_d = torch.linalg.vector_norm(kl_mask_rejected * rejected_loss_mask, ord=1, dim=-1)
            if self.l1_norm_param_d != 0:
                losses += self.l1_norm_param_d * (norm_loss_chosen_d + norm_loss_rejected_d)

        chosen_rewards = self.beta * (chosen_reward + chosen_token_kl)
        rejected_rewards = self.beta * (rejected_reward + rejected_token_kl)

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = ((policy_rejected_token_logps * rejected_loss_mask).sum(-1)).detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = ((policy_chosen_token_logps * chosen_loss_mask).sum(-1)).detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        metrics[f"{prefix}WgtSeqKL/chosen"] = -chosen_token_kl.detach().mean().cpu()
        metrics[f"{prefix}WgtSeqKL/rejected"] = -rejected_token_kl.detach().mean().cpu()
        metrics[f"{prefix}WgtSeqKL/margin"] = (chosen_token_kl.sum(-1) - rejected_token_kl.sum(-1)).detach().abs().mean().cpu()
        metrics[f"{prefix}SeqKL/chosen"] = -premask_chosen_token_kl.sum(-1).detach().mean().cpu()
        metrics[f"{prefix}SeqKL/rejected"] = -premask_rejected_token_kl.sum(-1).detach().mean().cpu()
        metrics[f"{prefix}SeqKL/margin"] = (premask_chosen_token_kl.sum(-1) - premask_rejected_token_kl.sum(-1)).detach().abs().mean().cpu()

        # Keep track of sparsity
        rw_chosen_mask_sparsity, rw_rejected_mask_sparsity = [], []
        kl_chosen_mask_sparsity, kl_rejected_mask_sparsity = [], []
        for batch_id in range(0, chosen_token_kl.size(0)):
            if self.mask_config.get('mask_on_reward',False):
                total_chosen = rw_mask_chosen[batch_id][chosen_loss_mask[batch_id] > 0].size(0)
                nonzero_chosen = rw_mask_chosen[batch_id][chosen_loss_mask[batch_id] > 0].nonzero().size(0)
                rw_chosen_mask_sparsity.append((total_chosen - nonzero_chosen)*100 / total_chosen)

                total_rejected = rw_mask_rejected[batch_id][rejected_loss_mask[batch_id] > 0].size(0)
                nonzero_rejected = rw_mask_rejected[batch_id][rejected_loss_mask[batch_id] > 0].nonzero().size(0)
                rw_rejected_mask_sparsity.append((total_rejected - nonzero_rejected)*100 / total_rejected)
            
            if self.mask_config.get('mask_on_kl',False):
                total_chosen = kl_mask_chosen[batch_id][chosen_loss_mask[batch_id] > 0].size(0)
                nonzero_chosen = kl_mask_chosen[batch_id][chosen_loss_mask[batch_id] > 0].nonzero().size(0)
                kl_chosen_mask_sparsity.append((total_chosen - nonzero_chosen)*100 / total_chosen)

                total_rejected = kl_mask_rejected[batch_id][rejected_loss_mask[batch_id] > 0].size(0)
                nonzero_rejected = kl_mask_rejected[batch_id][rejected_loss_mask[batch_id] > 0].nonzero().size(0)
                kl_rejected_mask_sparsity.append((total_rejected - nonzero_rejected)*100 / total_rejected)
        #
        if self.mask_config.get('mask_on_reward',False):
            metrics[f"{prefix}Weights/rw_chosen_sparsity"] = sum(rw_chosen_mask_sparsity)/len(rw_chosen_mask_sparsity)
            metrics[f"{prefix}Weights/rw_rejected_sparsity"] = sum(rw_rejected_mask_sparsity)/len(rw_rejected_mask_sparsity)
        if self.mask_config.get('mask_on_kl',False):
            metrics[f"{prefix}Weights/kl_chosen_sparsity"] = sum(kl_chosen_mask_sparsity)/len(kl_chosen_mask_sparsity)
            metrics[f"{prefix}Weights/kl_rejected_sparsity"] = sum(kl_rejected_mask_sparsity)/len(kl_rejected_mask_sparsity)

        # Masks histogram
        if self.mask_config.get('mask_on_reward',False):
            metrics[f"{prefix}Weights/rw_chosen_hist"] = rw_mask_chosen[chosen_loss_mask>0].detach().view(-1).cpu()
            metrics[f"{prefix}Weights/rw_rejected_hist"] = rw_mask_rejected[rejected_loss_mask>0].detach().view(-1).cpu()
        if self.mask_config.get('mask_on_kl',False):
            metrics[f"{prefix}Weights/kl_chosen_hist"] = kl_mask_chosen[chosen_loss_mask>0].detach().view(-1).cpu()
            metrics[f"{prefix}Weights/kl_rejected_hist"] = kl_mask_rejected[rejected_loss_mask>0].detach().view(-1).cpu()

        if self.mask_config.get('mask_on_reward',False):
            metrics[f"{prefix}Weights/u_chosen_l1_norm"] = norm_loss_chosen_u.detach().mean().cpu()
            metrics[f"{prefix}Weights/u_rejected_l1_norm"] = norm_loss_rejected_u.detach().mean().cpu()
        
        if self.mask_config.get('mask_on_kl',False):
            metrics[f"{prefix}Weights/d_chosen_l1_norm"] = norm_loss_chosen_d.detach().mean().cpu()
            metrics[f"{prefix}Weights/d_rejected_l1_norm"] = norm_loss_rejected_d.detach().mean().cpu()
        
        del (total_rejected, nonzero_rejected, rw_chosen_mask_sparsity, rw_rejected_mask_sparsity, kl_chosen_mask_sparsity, kl_rejected_mask_sparsity)
        torch.cuda.empty_cache()
        gc.collect()

        return losses.mean(), metrics


    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt):], ref[len(prompt):]]
                            for prompt, pol, ref in zip(
                                random_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            if "_hist" in key:
                continue
                # logs[key] = torch.hstack([x.view(1, -1) for x in metrics]).view(-1).tolist()
            else:
                logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "dpo" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            all_params = self.get_decay_parameter_names(opt_model)
            mask_decay_params = [x for x in all_params if "_mask" in x]
            decay_parameters = [x for x in all_params if x not in mask_decay_params]
        if self.optimizer is None:
            all_params = self.get_decay_parameter_names(opt_model)
            mask_decay_params = [x for x in all_params if "_mask" in x]
            decay_parameters = [x for x in all_params if x not in mask_decay_params]

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if
                        (n not in decay_parameters and n not in mask_decay_params and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in mask_decay_params and p.requires_grad)
                    ],
                    "weight_decay": self.args.mask_weight_decay,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    ###################################################
    # Analysis

    def get_analysis_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        from tools.utils import filter_pad_and_split
        
        metrics = {}

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if (
            "reference_chosen_logps" in batch
            and "reference_rejected_logps" in batch
            and self.args.rpo_alpha is not None
        ):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logits,
                            reference_rejected_logits,
                            _, _, _, _,
                            _, _, _, _,
                            ref_chosen_hidden_states,
                            ref_rejected_hidden_states
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logits,
                        reference_rejected_logits,
                        _, _, _, _,
                        _, _, _, _,
                        ref_chosen_hidden_states,
                        ref_rejected_hidden_states
                    ) = self.concatenated_forward(self.ref_model, batch)

        (
            policy_chosen_logits,
            policy_rejected_logits,
            rw_mask_chosen,rw_mask_rejected,
            kl_mask_chosen,kl_mask_rejected,
            chosen_loss_mask,
            rejected_loss_mask,
            chosen_labels,
            rejected_labels,
            pol_chosen_hidden_states, 
            pol_rejected_hidden_states
        ) = self.concatenated_forward(model, batch,
                                      ref_chosen_states=ref_chosen_hidden_states,
                                      ref_rejected_states=ref_rejected_hidden_states)

        # Start by computing mask * (log pi - log pi_r)
        policy_chosen_logps_vocab = policy_chosen_logits.log_softmax(-1)
        policy_rejected_logps_vocab = policy_rejected_logits.log_softmax(-1)
        reference_chosen_logps_vocab = reference_chosen_logits.log_softmax(-1)
        reference_rejected_logps_vocab = reference_rejected_logits.log_softmax(-1)

        policy_chosen_token_logps = torch.gather(policy_chosen_logps_vocab, dim=2, index=chosen_labels.unsqueeze(2)).squeeze(2)  # (b, seq)
        policy_rejected_token_logps = torch.gather(policy_rejected_logps_vocab, dim=2, index=rejected_labels.unsqueeze(2)).squeeze(2)  # (b, seq)
        reference_chosen_token_logps = torch.gather(reference_chosen_logps_vocab, dim=2, index=chosen_labels.unsqueeze(2)).squeeze(2)  # (b, seq)
        reference_rejected_token_logps = torch.gather(reference_rejected_logps_vocab, dim=2, index=rejected_labels.unsqueeze(2)).squeeze(2)  # (b, seq)

        chosen_reward = policy_chosen_token_logps - reference_chosen_token_logps
        rejected_reward = policy_rejected_token_logps - reference_rejected_token_logps

        pre_mask_reward_chosen = chosen_reward.clone()
        pre_mask_reward_rejected = rejected_reward.clone()

        if self.mask_config.get('mask_on_reward',False):
            chosen_reward = rw_mask_chosen * chosen_reward
            rejected_reward = rw_mask_rejected * rejected_reward

        post_mask_reward_chosen = chosen_reward.clone()
        post_mask_reward_rejected = rejected_reward.clone()

        chosen_reward = (chosen_reward * chosen_loss_mask).sum(-1)
        rejected_reward = (rejected_reward * rejected_loss_mask).sum(-1)  # (b,)
        u = chosen_reward - rejected_reward

        # Token-level KL divergences
        reference_chosen_ps = reference_chosen_logits.softmax(-1)  # (b, seq, vocab) distribution over vocab
        reference_rejected_ps = reference_rejected_logits.softmax(-1)

        chosen_token_kl = (reference_chosen_ps * (policy_chosen_logps_vocab - reference_chosen_logps_vocab)).sum(-1)  # (b, seq) sum across the vocab dist
        rejected_token_kl = (reference_rejected_ps * (policy_rejected_logps_vocab - reference_rejected_logps_vocab)).sum(-1)

        premask_chosen_token_kl = chosen_token_kl.clone()
        premask_rejected_token_kl = rejected_token_kl.clone()

        if self.mask_config.get('mask_on_kl',False):
            chosen_token_kl = kl_mask_chosen * chosen_token_kl
            rejected_token_kl = kl_mask_rejected * rejected_token_kl
        #

        postmask_chosen_token_kl = chosen_token_kl.clone()
        postmask_rejected_token_kl = rejected_token_kl.clone()

        chosen_token_kl = (chosen_token_kl * chosen_loss_mask).sum(-1)  # (b,)
        rejected_token_kl = (rejected_token_kl * rejected_loss_mask).sum(-1) # (b,)
        d = chosen_token_kl - rejected_token_kl

        #######
        # LOSS
        #######
        losses = -F.logsigmoid(self.beta * (u - d))

        norm_loss_chosen_u,norm_loss_rejected_u = 0,0
        norm_loss_chosen_d,norm_loss_rejected_d = 0,0

        if self.mask_config.get('mask_on_reward',False):
            norm_loss_chosen_u =  torch.linalg.vector_norm(rw_mask_chosen * chosen_loss_mask, ord=1, dim=-1)
            norm_loss_rejected_u = torch.linalg.vector_norm(rw_mask_rejected * rejected_loss_mask, ord=1, dim=-1)
            if self.l1_norm_param_u != 0:
                losses += self.l1_norm_param_u * (norm_loss_chosen_u + norm_loss_rejected_u)
        
        if self.mask_config.get('mask_on_kl',False):
            norm_loss_chosen_d = torch.linalg.vector_norm(kl_mask_chosen * chosen_loss_mask, ord=1, dim=-1)
            norm_loss_rejected_d = torch.linalg.vector_norm(kl_mask_rejected * rejected_loss_mask, ord=1, dim=-1)
            if self.l1_norm_param_d != 0:
                losses += self.l1_norm_param_d * (norm_loss_chosen_d + norm_loss_rejected_d)        

        dpo_chosen_rewards = self.beta * (chosen_reward + chosen_token_kl)
        dpo_rejected_rewards = self.beta * (rejected_reward + rejected_token_kl)

        dpo_reward_accuracies = (dpo_chosen_rewards > dpo_rejected_rewards).float()

        cmarg = ((reference_chosen_token_logps.to(torch.float32) - policy_chosen_token_logps.to(torch.float32)) * chosen_loss_mask).sum(-1)
        rmarg = ((reference_rejected_token_logps.to(torch.float32) - policy_rejected_token_logps.to(torch.float32)) * rejected_loss_mask).sum(-1)

        kl_pol_ref_chosen = - torch.exp(cmarg) - cmarg - 1.0
        kl_pol_ref_rejected = torch.exp(rmarg) - rmarg - 1.0

        metrics[f"dpo_rewards_chosen"] = dpo_chosen_rewards.cpu().numpy().tolist()
        metrics[f"dpo_rewards_rejected"] = dpo_rejected_rewards.cpu().numpy().tolist()
        metrics[f"dpo_rewards_accuracies"] = dpo_reward_accuracies.cpu().numpy().tolist()

        metrics["u_chosen_pre-mask"] = filter_pad_and_split(pre_mask_reward_chosen, chosen_loss_mask)
        metrics["u_rejected_pre-mask"] = filter_pad_and_split(pre_mask_reward_rejected, rejected_loss_mask)

        metrics["u_chosen_post-mask"] = filter_pad_and_split(post_mask_reward_chosen, chosen_loss_mask)
        metrics["u_rejected_post-mask"] = filter_pad_and_split(post_mask_reward_rejected, rejected_loss_mask)

        metrics["d_chosen_pre-mask"] = filter_pad_and_split(premask_chosen_token_kl, chosen_loss_mask)
        metrics["d_rejected_pre-mask"] = filter_pad_and_split(premask_rejected_token_kl, rejected_loss_mask)

        metrics["d_chosen_post-mask"] = filter_pad_and_split(postmask_chosen_token_kl, chosen_loss_mask)
        metrics["d_rejected_post-mask"] = filter_pad_and_split(postmask_rejected_token_kl, rejected_loss_mask)

        if self.mask_config.get('mask_on_reward',False):
            metrics["u_chosen_mask"] = filter_pad_and_split(rw_mask_chosen, chosen_loss_mask)
            metrics["u_rejected_mask"] = filter_pad_and_split(rw_mask_rejected, rejected_loss_mask)
        if self.mask_config.get('mask_on_kl',False):
            metrics["d_chosen_mask"] = filter_pad_and_split(kl_mask_chosen, chosen_loss_mask)
            metrics["d_rejected_mask"] = filter_pad_and_split(kl_mask_rejected, rejected_loss_mask)
        
        metrics[f"kl_pol_ref_chosen"] = kl_pol_ref_chosen.cpu().numpy().tolist()
        metrics[f"kl_pol_ref_rejected"] = kl_pol_ref_rejected.cpu().numpy().tolist()

        torch.cuda.empty_cache()
        gc.collect()

        return metrics
