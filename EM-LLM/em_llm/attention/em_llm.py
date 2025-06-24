import torch
from torch.nn import CrossEntropyLoss, MSELoss

from typing import List, Optional, Tuple, Union
from .context_manager import ContextManager

from transformers.modeling_outputs import CausalLMOutputWithPast

from .similarity_refinement import events_with_similarity_adjustment

def em_llm_attn_forward(
    model,
    n_local, 
    n_init, 
    max_block_size, 
    max_cached_block,
    exc_block_size, 
    repr_topk: int = 1,
    surprisal_threshold_gamma: float = 1.1,
    async_global_stream=True,
    pin_memory=False,
    perhead=False,
    n_mem: int = 2048,
    min_block_size: int = 1,
    block_similarity_topk: int = False,
    similarity_refinement_kwargs: dict = {},
    contiguity_buffer_kwargs: dict = {},
    random_topk_blocks=False,
    infini_attention=False,
    uniform_blocks=False,
    *args, **kwargs
):

    def forward(
        self, 
        query : torch.Tensor,
        key_value : torch.Tensor,
        position_bias : Optional[torch.Tensor],
        use_cache: bool,
        past_key_value,
        project_q, 
        project_k, 
        project_v, 
        attention_out, 
        dim_head, 
        num_heads, 
        num_heads_kv,
    ):
        
        batch_size = query.size(0)
        
        len_q = query.size(1)
        len_k = key_value.size(1)
        
        assert use_cache
        if project_k is not None:
            h_q = project_q(query)  # (batch, len_q, num_heads * dim_head)
            h_k = project_k(key_value)  # (batch, len_k, num_heads_kv * dim_head)
            h_v = project_v(key_value)  # (batch, len_k, num_heads_kv * dim_head)
        else:
            qkv = project_q(query)
            query_pos = num_heads * dim_head
            h_q = qkv[..., :query_pos]
            h_k = qkv[..., query_pos : query_pos + num_heads_kv * dim_head]
            h_v = qkv[..., query_pos + num_heads_kv * dim_head :]
        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
        
        if past_key_value is None:
            past_key_value = ContextManager(
                layer_idx=self.layer_idx,
                position_embedding=position_bias,
                n_init=n_init,
                n_local=n_local,
                max_block_size=max_block_size,
                max_cached_block=max_cached_block,
                exc_block_size=exc_block_size,
                min_block_size=min_block_size,
                async_global_stream=async_global_stream,
                pin_memory=pin_memory,
                perhead=perhead,
                repr_topk=repr_topk,
                surprisal_threshold_gamma=surprisal_threshold_gamma,
                n_mem=n_mem,
                block_similarity_topk=block_similarity_topk,
                uniform_blocks=uniform_blocks,
                random_topk_blocks=random_topk_blocks,
                infini_attention=infini_attention,
                **similarity_refinement_kwargs,
                **contiguity_buffer_kwargs,
                **kwargs
            )

        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v

        o = past_key_value.append(
            local_q, local_k, local_v,
            global_q, global_k, global_v,
        )

        o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
        o = o.reshape(batch_size, len_q, dim_head * num_heads)
        o = attention_out(o)

        return o, None, past_key_value

    return forward


def em_llm_causal_lm_forward(
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
    em_labels: Optional[bool] = None,
    **kwargs
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
        return_dict=return_dict
    )

    hidden_states = outputs[0]
    past_key_values = outputs[1]

    logits = self.lm_head(hidden_states)
    if past_key_values[0].use_hf_acc:
        logits = logits.to(torch.cuda.current_device())
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

    if em_labels is not None:
        if em_labels.dtype != torch.bool:
            # compute surprisal of given outcome
            prob = torch.softmax(logits, dim = -1)
            surprisal = -torch.log(torch.gather(prob, dim=-1, index=em_labels.unsqueeze(-1))).squeeze(-1)
        else:
            surprisal = em_labels
    else:
        surprisal = None

    torch.cuda.synchronize()
    # update past_key_values
    surprisal_values = None
    if surprisal is not None and em_labels.dtype != torch.bool and not past_key_values[0].uniform_blocks:
        assert surprisal.shape == (past_key_values[0].batch_size, input_ids.shape[-1]), f'Problem with surprisal shape: {surprisal.shape}'
        if past_key_values[0].similarity_refinement:
            exc_length = input_ids.shape[-1]
            global_remainder_ed = past_key_values[0]._global_remainder_ed + exc_length
            global_remainder_len = global_remainder_ed - past_key_values[0]._global_remainder_st

            if global_remainder_ed <= exc_length:
                divide = surprisal > past_key_values[0].surprisal_threshold_gamma \
                * torch.std(surprisal, dim=-1) + torch.mean(surprisal, dim=-1)
            else:
                avg_st = max(global_remainder_ed - past_key_values[0].n_local, 0)
                avg_ed = max(global_remainder_ed - exc_length, past_key_values[0].n_init)
                divide = surprisal > past_key_values[0].surprisal_threshold_gamma \
                    * torch.std(past_key_values[0].global_remainder_surprisal[:, avg_st:avg_ed], dim=-1) \
                    + torch.mean(past_key_values[0].global_remainder_surprisal[:, avg_st:avg_ed], dim=-1)
            if global_remainder_len >= 2*exc_length and past_key_values[0].refine_with_buffer:
                # find the indices of the last event from the previous chunk for each batch
                last_divide = torch.zeros(past_key_values[0].batch_size)
                for u in range(past_key_values[0].batch_size):
                    last_events = torch.where(past_key_values[0].global_block_divide[u, global_remainder_ed-2*exc_length:global_remainder_ed-exc_length] > 0)[0]
                    if len(last_events) == 0:
                        last_divide[u] = exc_length
                    else:
                        last_divide[u] = last_events[-1]
                offsets = exc_length - last_divide.int()
                max_offset = torch.max(offsets)
            else:
                offsets = torch.zeros(past_key_values[0].batch_size).int()
                max_offset = 0
            st_layer = past_key_values[0].refine_from_layer                
            assert len(past_key_values) >= st_layer + 1, f'Problem with refine_from_layer param: {st_layer + 1} < {len(past_key_values)}'
            K = torch.clone(past_key_values[st_layer].global_remainder[0][:, :, global_remainder_ed-exc_length-max_offset:global_remainder_ed, :]).unsqueeze(dim=1).to(torch.cuda.current_device())
            for l in range(st_layer+1, len(past_key_values)):
                K = torch.cat((K, torch.clone(past_key_values[l].global_remainder[0][:, :, global_remainder_ed-exc_length-max_offset:global_remainder_ed, :]).unsqueeze(dim=1).to(torch.cuda.current_device())), dim=1)
            stacked_A = torch.einsum('blhtd,blhTd->btT', K, K).detach()
            del K
            # try casting A to GPU
            try:
                stacked_A = stacked_A.to(past_key_values[0].global_remainder[0].device)  # currently already on this device
            except Exception as e:
                print(f'Tried casting stacked_A to GPU, but failed with error: {e}')
            for u in range(past_key_values[0].batch_size):
                events_sur = torch.where(divide[u] > 0)[0]
                if len(events_sur) > 0:
                    events_sur_mod = events_with_similarity_adjustment(events_sur
                                                                        , stacked_A[u][max_offset-offsets[u]:, :][:, max_offset-offsets[u]:]
                                                                        , similarity_metric=past_key_values[0].similarity_metric
                                                                        , min_size=past_key_values[0].min_block_size
                                                                        , offset=offsets[u])
                    divide[u] = torch.zeros_like(divide[u])
                    divide[u][events_sur_mod] = True
            del stacked_A
            surprisal_values = torch.clone(surprisal)
            surprisal = divide
            assert surprisal.dtype == torch.bool, f'Problem with surprisal dtype after refinement: {surprisal.dtype}'
        
    for pkv in past_key_values:
        pkv.update_memory(input_ids.shape[-1], surprisal, surprisal_values=surprisal_values)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    