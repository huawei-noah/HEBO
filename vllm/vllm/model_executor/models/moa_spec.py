import copy
from typing import Iterable, List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import repeat_kv

import vllm.attention.backends.flash_attn
from vllm.attention import Attention
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.compressed_tensors.utils import get_compressed_tensors_cache_scale
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, Sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaMLP
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.moa_spec import MOASpecConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.vllm_flash_attn import flash_attn_with_kvcache


class FullAttention(Attention):

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata,
    ) -> torch.Tensor:
        nb_layers, nb_tokens = attn_metadata

        query = query.view(nb_tokens, nb_layers, self.impl.num_heads, self.impl.head_size).transpose(1, 2)
        key = key.view(nb_tokens, nb_layers, self.impl.num_kv_heads, self.impl.head_size).transpose(1, 2)
        value = value.view(nb_tokens, nb_layers, self.impl.num_kv_heads, self.impl.head_size).transpose(1, 2)

        key = repeat_kv(key, self.impl.num_queries_per_kv)
        value = repeat_kv(value, self.impl.num_queries_per_kv)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(nb_tokens, nb_layers, -1)
        return attn_output

class LlamaCrossAttention(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
        )

        self.empty_kv_cache = torch.empty(0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states2: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # qkv, _ = self.qkv_proj(hidden_states)
        # q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if hidden_states2 is None:
            q = torch.nn.functional.linear(hidden_states, self.qkv_proj.weight[: self.qkv_proj.output_partition_sizes[0]])

            q, _ = self.rotary_emb(positions, q, torch.zeros_like(q))
            num_tokens, hidden_size = q.shape
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]

            decode_query = q.view(-1, self.attn.impl.num_heads, self.attn.impl.head_size)

            if isinstance(self.attn.impl, vllm.attention.backends.flash_attn.FlashAttentionImpl):
                decode_output = flash_attn_with_kvcache(
                    q=decode_query.unsqueeze(1),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    block_table=attn_metadata.decode_metadata.block_tables,
                    cache_seqlens=attn_metadata.decode_metadata.seq_lens_tensor,
                    softmax_scale=self.attn.impl.scale,
                    causal=True,
                    alibi_slopes=None,
                    softcap=0,
                ).squeeze(1)
            else:
                key_cache, value_cache = PagedAttention.split_kv_cache(
                    kv_cache, self.attn.impl.num_kv_heads, self.attn.impl.head_size)

                decode_output = PagedAttention.forward_decode(
                    decode_query,
                    key_cache,
                    value_cache,
                    attn_metadata.decode_metadata.block_tables,
                    attn_metadata.decode_metadata.seq_lens_tensor,
                    attn_metadata.decode_metadata.max_decode_seq_len,
                    self.attn.impl.kv_cache_dtype,
                    self.attn.impl.num_kv_heads,
                    self.attn.impl.scale,
                    self.attn.impl.alibi_slopes,
                    1.0,
                    1.0,
                )
                decode_output = decode_output.view(-1, self.attn.impl.num_heads * self.attn.impl.head_size)

            attn_output = decode_output.view(attn_metadata.num_decode_tokens, hidden_size)
        else:
            q = torch.nn.functional.linear(hidden_states, self.qkv_proj.weight[: self.qkv_proj.output_partition_sizes[0]])
            kv = torch.nn.functional.linear(hidden_states2, self.qkv_proj.weight[self.qkv_proj.output_partition_sizes[0]:])
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)

            q, k = self.rotary_emb(positions, q, k)

            attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

        output, _ = self.o_proj(attn_output)
        return output

    def store_kv(self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata
    ):
        kv = torch.nn.functional.linear(hidden_states, self.qkv_proj.weight[self.qkv_proj.output_partition_sizes[0]:])
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)

        _, k = self.rotary_emb(positions, torch.zeros_like(k), k)

        key = k.view(-1, self.attn.impl.num_kv_heads, self.attn.impl.head_size)
        value = v.view(-1, self.attn.impl.num_kv_heads, self.attn.impl.head_size)

        if isinstance(self.attn.impl, vllm.attention.backends.flash_attn.FlashAttentionImpl):
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                kv_cache[0],
                kv_cache[1],
                attn_metadata.slot_mapping.flatten(),
                self.attn.kv_cache_dtype,
                1.0,
                1.0,
            )
        else:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.attn.impl.num_kv_heads, self.attn.impl.head_size
            )

            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache,
                                                attn_metadata.slot_mapping,
                                                self.attn.kv_cache_dtype,
                                                1.0, 1.0)


class LlamaCrossAttDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaCrossAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states2: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            if hidden_states2 is not None:
                hidden_states2 = self.input_layernorm(hidden_states2)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
            raise()
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states,
                                       hidden_states2=hidden_states2,
                                       kv_cache=kv_cache,
                                       attn_metadata=attn_metadata)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def store_kv(self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        hidden_states = self.input_layernorm(hidden_states)
        self.self_attn.store_kv(positions=positions,
                                        hidden_states=hidden_states,
                                        kv_cache=kv_cache,
                                        attn_metadata=attn_metadata)


class MOASpec(nn.Module):
    """This class implements the Mixture of Attention draft model from the paper: https://arxiv.org/pdf/2401.15077
    Reference implementation: https://github.com/SafeAILab/EAGLE
    
    Differences from reference implementation:
    1. In reference, LlamaDecoderLayer implementation doesn't have 
       input_layernorm for 1st decoder layer (https://github.com/SafeAILab/EAGLE/blob/7d065d084443fbfd386f88839efd7193c12be869/eagle/model/cnets.py#L427) 
       but we do as HF implementation also does.
    2. We allow any decoder layer to be used in EAGLE whereas in reference 
       decoder layer is fixed to be LlamaDecoderLayer.
    3. We have an optional token_map which reduces draft vocab to most 
       frequently used tokens to give some additional speed-up by reducing 
       sampling overhead. This is disabled unless the checkpoint file has 
       explicit token_map tensor and config has an optional attribute 
       truncated_vocab_size < vocab_size. To use this technique, one has to find
       the top-k most frequent tokens in target dataset and add that as a tensor
       in the draft checkpoint (using key token_map). Also, the draft config
       needs to have truncated_vocab_size (=k) as an attribute."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    def __init__(self, config: MOASpecConfig, *args, **kwargs) -> None:
        super().__init__()
        self.config = config

        if not self.config.disable_LSA:
            self.layer_self_attention = LlamaDecoderLayer(self.config.layer_self_attention, *args, **kwargs)
            # inject full attention class
            self.layer_self_attention.self_attn.attn.__class__ = FullAttention
            self.Ekv2E = nn.Linear(config.kv_hidden_size, config.hidden_size)

        self.self_attention = LlamaDecoderLayer(self.config.self_attention, *args, **kwargs)
        self.cross_attention = LlamaCrossAttDecoderLayer(self.config.cross_attention, *args, **kwargs)

        self.orig_vocab_size = config.vocab_size
        self.truncated_vocab_size = config.truncated_vocab_size
        self.unpadded_vocab_size = self.truncated_vocab_size

#        self.lm_head = ParallelLMHead(
#            self.unpadded_vocab_size,
#            config.hidden_size,
#            org_num_embeddings=self.truncated_vocab_size,
#            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
#        )

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                self.truncated_vocab_size,
                                                logit_scale)

        # Token map is a idx to token mapping to reduce the vocab size for
        # the draft model. Using smaller vocab size for draft, containing
        # only most frequent tokens reduces the speculation overhead. This
        # doesn't affect the acceptance rate much and thus gives more speed
        # -up. By default, this is disabled and is only used if the
        # checkpoint file has token_map tensor.
        self.token_map = None

        self.sampler = Sampler()

    def share_weights(self, model):
        self.lm_head = model.lm_head
        self.embed_tokens = model.model.embed_tokens
        if not self.config.disable_LSA:
            model.post_process_KV = self.post_process_KV

    def post_process_KV(self, KV):
        nb_token, nb_layer = KV.shape[0], KV.shape[1]
        positions = torch.arange(nb_layer, device=KV.device)[None].repeat(nb_token, 1)

        hidden_states, residual = self.layer_self_attention(
            hidden_states=KV.view(-1, KV.shape[-1]),
            positions=positions.view(-1),
            kv_cache=None,
            attn_metadata=(nb_layer, nb_token),
            residual=None,
        )

        hidden_states = hidden_states + residual.view(nb_token, nb_layer, -1)

        return self.Ekv2E(hidden_states.mean(1))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        previous_hidden_states: torch.Tensor = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        attn_metadata_ca: AttentionMetadata = None,
        previous_positions: torch.Tensor = None,
    ) -> torch.Tensor:
        tok_embeds = self.embed_tokens(input_ids)

        hidden_states, residual = self.self_attention(
            positions=positions,
            hidden_states=tok_embeds,
            kv_cache=kv_caches[0],
            attn_metadata=attn_metadata,
            residual=None)

        hidden_states += residual

        # on first prefill, no decoding from drafter after that
        if attn_metadata.prefill_metadata is not None:
            self.cross_attention.store_kv(positions=positions,
                                          hidden_states=previous_hidden_states,
                                          kv_cache=kv_caches[1],
                                          attn_metadata=attn_metadata)
        else:
            # on later stage

            # just after verification
            if previous_hidden_states is not None and previous_positions is not None:
                self.cross_attention.store_kv(positions=previous_positions,
                                              hidden_states=torch.cat(previous_hidden_states, 0) if isinstance(previous_hidden_states, list) else previous_hidden_states,
                                              kv_cache=kv_caches[1],
                                              attn_metadata=attn_metadata_ca)

            hidden_states, residual = self.cross_attention(
                positions=positions,
                hidden_states=hidden_states,
                hidden_states2=None,
                kv_cache=kv_caches[1],
                attn_metadata=attn_metadata_ca,
                residual=None)

            hidden_states += residual

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)

        if self.token_map is not None:
            _logits = logits
            logits = -torch.inf * torch.ones(
                size=(*_logits.shape[:-1], self.orig_vocab_size),
                device=_logits.device,
                dtype=_logits.dtype)

            logits[..., self.token_map] = _logits

        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if scale_name := get_compressed_tensors_cache_scale(name):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
