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
# ============================================================================

from ..masks import (
    SimpleMaskLayer,
    DoubleFFMaskLayer,
    SimpleMaskAllLayers,
    DoubleFFMaskAllLayers,
    RandomMask,
    BinaryMaskLast,
    BinaryMaskAllLayers,
    PipelineWrapNL,
    PipelineWrapCode,
    PipelineWrapNLGPT2,
    PipelineWrapNLGPTJ,
)

from ..masks.modeling_gpt_neox import GPTNeoXForCausalLM
from ..masks.modeling_gpt_bigcode import GPTBigCodeForCausalLM
from ..masks.modeling_gpt2 import GPT2LMHeadModel
from ..masks.modeling_gptj import GPTJForCausalLM


def filter_pad_and_split(tensor, mask, padding_id=0):
    assert tensor.size() == mask.size(), f"{tensor.size()} <> {mask.size()}"

    if tensor is None:
        return None

    extracted = []
    bs = tensor.size(0)
    for i in range(bs):
        nptensor = tensor[i][mask[i,:]!=padding_id].cpu().numpy().tolist()
        extracted.append(nptensor)
    return extracted


def get_sparse_pipeline(model_name, config, args, model_kwargs):
    #########################
    # Instantiate pipeline with base model wgts
    #########################
    mask_model_class = None
    if args.mask_model == "simple_last":
        mask_model_class = SimpleMaskLayer
    elif args.mask_model == "ml_last":
        mask_model_class = DoubleFFMaskLayer
    elif args.mask_model == "ml_all":
        mask_model_class = DoubleFFMaskAllLayers
    elif args.mask_model == "simple_all":
        mask_model_class = SimpleMaskAllLayers
    elif args.mask_model == "random":
        mask_model_class = RandomMask
    elif args.mask_model == "binary_last":
        mask_model_class = BinaryMaskLast
    elif args.mask_model == "binary_all":
        mask_model_class = BinaryMaskAllLayers
    
    common, rw_mask, kl_mask = None, None, None
    if not args.rw_kl_independent:
        common = mask_model_class(config, drpt=args.mask_dropout,
                                  layer_activation=args.mask_layer_activation,
                                  mixer_activation=args.mask_mixer_activation)
    else:
        if args.mask_on_reward:
            rw_mask = mask_model_class(config, drpt=args.mask_dropout,
                                       layer_activation=args.mask_layer_activation,
                                       mixer_activation=args.mask_mixer_activation)
        if args.mask_on_kl:
            kl_mask = mask_model_class(config, drpt=args.mask_dropout,
                                       layer_activation=args.mask_layer_activation,
                                       mixer_activation=args.mask_mixer_activation)

    mask_config = {
        'mask_model': args.mask_model,
        'mask_on_reward': args.mask_on_reward,
        'mask_on_kl': args.mask_on_kl,
        'is_rw_kl_independent': args.rw_kl_independent
    }

    if config.model_type == "gptj":
        pipeline_cls = PipelineWrapNLGPTJ

    elif config.model_type == 'gpt_bigcode':    
        pipeline_cls = PipelineWrapCode
    
    elif config.model_type == "gpt2":
        pipeline_cls = PipelineWrapNLGPT2

    else:
        pipeline_cls = PipelineWrapNL

    pipe = pipeline_cls.from_pretrained(
        model_name,
        rw_mask=rw_mask, kl_mask=kl_mask, common=common,
        mask_config=mask_config,
        **model_kwargs
    )

    return pipe


def get_mapo_model(model_name, architectures, model_kwargs):
    #########################
    # Instantiate pipeline with base model wgts
    #########################
    for arch in architectures:
        if arch == "GPTNeoXForCausalLM":
            return GPTNeoXForCausalLM.from_pretrained(model_name, **model_kwargs)
        elif arch == "GPTBigCodeForCausalLM":
            return GPTBigCodeForCausalLM.from_pretrained(model_name, **model_kwargs)
        elif arch == "GPT2LMHeadModel":
            return GPT2LMHeadModel.from_pretrained(model_name, **model_kwargs)
        elif arch == "GPTJForCausalLM":
            return GPTJForCausalLM.from_pretrained(model_name, **model_kwargs)