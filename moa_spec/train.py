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

import logging

import hydra
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer

from moa_spec.utils import setup_distributed_environment, set_and_print_config, set_seed, prepare_dataset

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs/training/", config_name="default_training")
def main(cfg: DictConfig) -> None:

    local_rank, world_size, seed_gpu_wise, device_map = setup_distributed_environment(cfg.seed)

    set_and_print_config(cfg, rank=local_rank)

    set_seed(cfg.seed)

    model_kwargs = hydra.utils.instantiate(cfg.model_kwargs)
    model_class = hydra.utils.instantiate(cfg.method.model_class)

    model_config = hydra.utils.instantiate(cfg.method.model_config) if hasattr(cfg.method, "model_config") else {}

    model = model_class.from_pretrained(
        **model_kwargs,
        **model_config,
        device_map=device_map
    )

    if cfg.drafter is not None:
        model.custom_load(cfg.drafter)

    # the Trainer detects the trainable parameters through requires_grad
    for param in model.parameters():
        param.requires_grad = False

    if cfg.method.name == "moa_spec":
        param_groups = (model.self_attention.parameters(),
                        model.cross_attention.parameters(),
                        model.layer_self_attention.parameters(),
                        model.Ekv2E.parameters())
    elif cfg.method.name == "eagle":
        param_groups = (model.spec_model.parameters(),
                        model.spec_model_fc.parameters())
    elif cfg.method.name == "independent":
        param_groups = (model.spec_model.model.layers.parameters(), )
    else:
        raise()

    for param_group in param_groups:
        for param in param_group:
            param.requires_grad = True

    tokenizer_kwargs = hydra.utils.instantiate(cfg.tokenizer_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
    assert tokenizer.padding_side == "right", "SFT/distillation works betting with padding right"
    if tokenizer.pad_token is None:
        pad_token = tokenizer.convert_ids_to_tokens(model.lm_head.out_features - 1)
        tokenizer.pad_token = pad_token
        if local_rank == 0:
            logger.warning(f"Padding token not set, setting it to {pad_token}")

    # use a different seed per GPU
    set_seed(seed_gpu_wise)

    training_dataset = load_dataset(**cfg.training_dataset)
    validation_dataset = load_dataset(**cfg.validation_dataset)

    training_dataset = prepare_dataset(
        training_dataset,
        tokenizer,
        total_number_tokens=model.lm_head.out_features,
        rank=local_rank,
        max_length=cfg.max_length
    )

    validation_dataset = prepare_dataset(
        validation_dataset,
        tokenizer,
        total_number_tokens=model.lm_head.out_features,
        rank=local_rank,
        max_length=cfg.max_length
    )

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        model=model,
        tokenizer=tokenizer,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        rank=local_rank,
        world_size=world_size
    )

    trainer.train()

if __name__ == "__main__":
    main()
