import os
from typing import Optional, Union

from transformers import AutoConfig, PretrainedConfig


class MOASpecConfig(PretrainedConfig):
    model_type = "moa_spec"

    def __init__(self,
                hidden_size: int = 4096,
                kv_hidden_size: int = 1024,
                num_attention_heads: int = 32,
                layer_self_attention_num_key_value_heads: int = 16,
                layer_self_attention_intermediate_size: int = 6144,
                self_attention_num_key_value_heads: int = 8,
                self_attention_intermediate_size: int = 512,
                cross_attention_num_key_value_heads: int = 8,
                cross_attention_intermediate_size: int = 7168,
                hidden_act: str = "silu",
                rms_norm_eps: float = 1e-05,
                rope_scaling = None,
                rope_theta: float = 500000.0,
                vocab_size: int = 128256,
                disable_LSA: bool = False,
                 **kwargs):

        self.disable_LSA = disable_LSA
        if not self.disable_LSA:
            self.layer_self_attention = PretrainedConfig()
            self.layer_self_attention.hidden_size = kv_hidden_size
            self.layer_self_attention.num_attention_heads = num_attention_heads
            self.layer_self_attention.num_key_value_heads = layer_self_attention_num_key_value_heads
            self.layer_self_attention.intermediate_size = layer_self_attention_intermediate_size
            self.layer_self_attention.hidden_act = hidden_act
            self.layer_self_attention.rms_norm_eps = rms_norm_eps
            self.layer_self_attention.rope_scaling = rope_scaling
            self.layer_self_attention.rope_theta = rope_theta
 
        self.self_attention = PretrainedConfig()
        self.self_attention.hidden_size = hidden_size
        self.self_attention.num_attention_heads = num_attention_heads
        self.self_attention.num_key_value_heads = self_attention_num_key_value_heads
        self.self_attention.intermediate_size = self_attention_intermediate_size
        self.self_attention.hidden_act = hidden_act
        self.self_attention.rms_norm_eps = rms_norm_eps
        self.self_attention.rope_scaling = rope_scaling
        self.self_attention.rope_theta = rope_theta
 
        self.cross_attention = PretrainedConfig()
        self.cross_attention.hidden_size = hidden_size
        self.cross_attention.num_attention_heads = num_attention_heads
        self.cross_attention.num_key_value_heads = cross_attention_num_key_value_heads
        self.cross_attention.intermediate_size = cross_attention_intermediate_size
        self.cross_attention.hidden_act = hidden_act
        self.cross_attention.rms_norm_eps = rms_norm_eps
        self.cross_attention.rope_scaling = rope_scaling
        self.cross_attention.rope_theta = rope_theta

        assert self_attention_num_key_value_heads == cross_attention_num_key_value_heads, \
            (f"Do not support different kv_caches sizes for SA and CA: "
             f"{self_attention_num_key_value_heads=} {cross_attention_num_key_value_heads=}")
        self.num_attention_heads = self_attention_num_key_value_heads
        # self.head_dim = hidden_size // self_attention_num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_hidden_layers = 2  # KV cache only needed on self_attention and cross_attention

        self.hidden_size = hidden_size
        self.kv_hidden_size = kv_hidden_size
        self.vocab_size = vocab_size
        self.truncated_vocab_size = vocab_size

        if "architectures" not in kwargs:
            kwargs["architectures"] = ["MOASpecModel"]

        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "MOASpecConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)
