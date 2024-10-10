import torch
from torch.nn import CrossEntropyLoss

from ..attention import RotaryEmbeddingESM, ATTN_FORWARD, CAUSAL_LM_FORWARD
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from typing import List, Optional, Tuple, Union
import math

def huggingface_forward(forward):
    def hf_forward(
        self,
        hidden_states: torch.Tensor,
        position_ids = None,
        past_key_value = None,
        use_cache: bool = False,
        **kwargs,
    ):
        if hasattr(self, 'q_proj'):
            q_proj = self.q_proj
            k_proj = self.k_proj
            v_proj = self.v_proj
        elif hasattr(self, 'qkv_proj'):
            q_proj = self.qkv_proj
            k_proj = None
            v_proj = None
        else:
            raise NotImplementedError(f"The attention module {self.__class__.__name__} does not appear to have the required projection methods.")
        hidden_states, loss, pkv = forward(
            self, 
            hidden_states, 
            hidden_states,
            position_ids, 
            use_cache, 
            past_key_value,
            q_proj, 
            k_proj, 
            v_proj, 
            self.o_proj, 
            self.head_dim, 
            self.num_heads, 
            self.num_key_value_heads,
        )

        return hidden_states, loss, pkv

    return hf_forward

def model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask = None,
    past_key_values = None,
    inputs_embeds = None,
    use_cache = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
    **kwargs
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is None and inputs_embeds is None:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
        if hasattr(self, "config") and hasattr(self.config, "scale_emb"):
            inputs_embeds = inputs_embeds * self.config.scale_emb

    if use_cache:
        pkv = tuple()
    else:
        pkv = None
        

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for i, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=self.position_bias,
            past_key_value=past_key_values[i] if past_key_values is not None else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            _cache = layer_outputs[2 if output_attentions else 1]
            pkv = pkv + (_cache,)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, pkv, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=pkv,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def causal_lm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if labels.shape[-1] != logits.shape[-2]:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            else:
                shift_logits = logits
                shift_labels = labels
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def patch_hf(
    model,
    attn_type: str = "em_llm",
    attn_kwargs: dict = {},
    base = None, 
    distance_scale = None,
    **kwargs
):
    attn_kwargs.update(kwargs)
        
    # This approach lacks scalability and will be refactored.
    from transformers import LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Phi3ForCausalLM

    forward = huggingface_forward(ATTN_FORWARD[attn_type](model=model, **attn_kwargs))

    if isinstance(model, LlamaForCausalLM) or isinstance(model, MistralForCausalLM) \
        or isinstance(model, Qwen2ForCausalLM) or isinstance(model, Phi3ForCausalLM) \
        or model.__class__.__name__ == "Phi3ForCausalLM" \
        or model.__class__.__name__ == "MiniCPMForCausalLM":
        Attention = model.model.layers[0].self_attn.__class__
        Model = model.model.__class__
    else:
        raise ValueError(f"Only supports llama, mistral, phi3, and qwen2 models. {model.__class__.__name__} was passed.")

    hf_rope = model.model.layers[0].self_attn.rotary_emb 
    if not hasattr(hf_rope, 'dim'):
        hf_rope.dim = hf_rope.rope_kwargs.get("dim", None)
        hf_rope.base = hf_rope.rope_kwargs.get("base", None).rope_theta
        if hf_rope.dim is None and hf_rope.config is not None:
            hf_rope.base = hf_rope.config.rope_theta
            partial_rotary_factor = hf_rope.config.partial_rotary_factor if hasattr(hf_rope.config, "partial_rotary_factor") else 1.0
            head_dim = getattr(hf_rope.config, "head_dim", hf_rope.config.hidden_size // hf_rope.config.num_attention_heads)
            hf_rope.dim = int(head_dim * partial_rotary_factor)
        else: 
            raise NotImplementedError 
    base = base if base is not None else hf_rope.base
    distance_scale = distance_scale if distance_scale is not None else 1.0
    if hasattr(hf_rope, 'short_factor'):
        new_max_pos_emb = attn_kwargs["n_local"] + attn_kwargs["exc_block_size"]
        scale = new_max_pos_emb / hf_rope.original_max_position_embeddings
        if scale <= 1.0:
            ext_factors = torch.tensor(hf_rope.short_factor)
        else:
            print(f"Extending context past original window with scale factor: {scale}")
            ext_factors = torch.tensor(hf_rope.long_factor)
            distance_scale = math.sqrt(1 + math.log(scale) / math.log(hf_rope.original_max_position_embeddings))
    else:
        ext_factors = torch.tensor(1.0)
    rope = RotaryEmbeddingESM(
        hf_rope.dim,
        base,
        distance_scale,
        ext_factors
    )
    model.model.position_bias = rope

    def set_forward(m):
        if isinstance(m, Attention):
            m._old_forward = m.forward
            m.forward = forward.__get__(m, Attention)

    model.apply(set_forward)

    model.model._old_forward = model.model.forward
    model.model.forward = model_forward.__get__(model.model, Model)

    model._old_forward = model.model.forward
    if attn_type in CAUSAL_LM_FORWARD:
        model.forward = CAUSAL_LM_FORWARD[attn_type].__get__(model, model.__class__)
    else:
        model.forward = causal_lm_forward.__get__(model, model.__class__)

    return model

