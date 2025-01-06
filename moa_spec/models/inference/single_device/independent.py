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

import warnings
from typing import Tuple, Union, Optional, List

import numpy as np
import torch
from transformers import LlamaForCausalLM, LogitsProcessorList, StoppingCriteriaList, DynamicCache, AutoModelForCausalLM
from transformers.generation import GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput, validate_stopping_criteria
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateNonBeamOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from moa_spec.models.inference.tree_attention import LlamaModelTreeAttentionMask


class IndependentLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self,
                 config,
                 drafter,
                 tree_decoding,
                 verification,
                 depth,
                 breadth,
                 total_tokens,
        ):
        super().__init__(config)
        self.call_to_big = 0
        self.drafter_kwargs = drafter
        self.spec_model = None

        if not isinstance(self.model, LlamaModelTreeAttentionMask):
            self.model.__class__ = LlamaModelTreeAttentionMask

        self.stats = []
        self.tree_decoding = tree_decoding
        self.verification = verification
        self.depth = depth
        self.breadth = breadth
        self.total_tokens = total_tokens
        self.call_to_big += 1
        assert self.dtype == next(self.parameters()).dtype
        self.min_dtype = torch.finfo(self.dtype).min

        if self.depth == 1 and self.verification:
            self.verification = False

        if self.tree_decoding:
            assert self.verification

            init_tree_mask = torch.full(
                (breadth, breadth),
                fill_value=self.min_dtype,
                dtype=self.dtype,
            )
            init_tree_mask.fill_diagonal_(0)
            self.init_tree_mask = init_tree_mask[None, None]
        else:
            self.breadth = 1

    def __del__(self):
        if len(self.stats) > 0:
            m_value = max(self.stats)
            prob = np.histogram(self.stats, bins=np.arange(1, m_value + 2), density=True)
            print(f"{np.histogram(self.stats, bins=np.arange(1, m_value + 2))}")
            print(f"{prob}")
            print(f"avg len: {prob[0] @ np.arange(1, m_value + 2)[:prob[0].shape[0]]}")

    def custom_load(self, load_path, dtype):
        self.spec_model = AutoModelForCausalLM.from_pretrained(
            **self.drafter_kwargs,
            device_map=next(self.parameters()).device,
        )
        print(f"LOADING successfully {load_path} states")

        if not isinstance(self.spec_model.model, LlamaModelTreeAttentionMask):
            self.spec_model.model.__class__ = LlamaModelTreeAttentionMask

    def forward(
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
        cache_position: Optional[torch.LongTensor] = None,
        last_hidden_state=None,
        with_next_feat=False,
        response_masks=None,
        recall_big=False,
        past_key_values2=None,
        gen=False,
        tree_masks=None,
        gen_step=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if last_hidden_state is None:
            self.call_to_big += 1
            with torch.no_grad():
                base_model_output = super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                                    position_ids=position_ids,
                                                    past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                                                    labels=labels,
                                                    use_cache=True, output_attentions=output_attentions,
                                                    output_hidden_states=True, return_dict=return_dict,
                                                    cache_position=cache_position, **kwargs)

            input_embs = base_model_output.hidden_states[0]
            last_hidden_state = base_model_output.hidden_states[-1]

            past_key_values2 = DynamicCache()
            base_model_output["hidden_states"] = [input_embs]
            base_model_output["logits"] = self.lm_head(last_hidden_state)
            base_model_output["past_key_values2"] = past_key_values2
            return base_model_output

        if recall_big:
            self.call_to_big += 1
            with torch.no_grad():
                base_model_output = super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                                    position_ids=position_ids,
                                                    past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                                                    labels=labels,
                                                    use_cache=True, output_attentions=output_attentions,
                                                    output_hidden_states=True, return_dict=return_dict,
                                                    cache_position=cache_position, **kwargs)

                new_last_hidden_state = base_model_output.hidden_states[-1]

            base_model_output["hidden_states"] = last_hidden_state
            base_model_output["logits"] = self.lm_head(new_last_hidden_state)

            return base_model_output

        if past_key_values2.get_seq_length() == 0:
            input_embeds = torch.cat((last_hidden_state[0], self.model.embed_tokens(input_ids)), 1)
            position_ids2 = torch.arange(input_embeds.shape[1], device=input_ids.device)[None]

            causal_mask = torch.full(
                (input_embeds.shape[1], input_embeds.shape[1]),
                fill_value=self.min_dtype,
                dtype=self.dtype,
                device=input_ids.device,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask[None, None]

            outputs = self.spec_model.model(
                inputs_embeds=input_embeds,
                position_ids=position_ids2,
                attention_mask=causal_mask,
                use_cache=True,
                past_key_values=past_key_values2
            )

            base_model_output = {}
            base_model_output["hidden_states"] = [input_embeds]
            attn_output = outputs["last_hidden_state"][:, -1:]
        else:
            position_ids2 = cache_position[None]

            if gen_step == 1:  # comes in only after recall_big=True
                input_embs = self.model.embed_tokens(input_ids)
                position_ids2 = position_ids2 - torch.arange(input_ids.shape[1] - 1, -1, -1, device=input_ids.device)

                tree_masks = torch.full(
                    (past_key_values2.get_seq_length() + input_embs.shape[1],
                     past_key_values2.get_seq_length() + input_embs.shape[1]),
                    fill_value=self.min_dtype,
                    dtype=self.dtype,
                    device=self.device,
                )
                tree_masks = torch.triu(tree_masks, diagonal=1)[None, None, past_key_values2.get_seq_length():]
            else:
                input_embs = self.model.embed_tokens(input_ids[:, -self.breadth:])

            outputs = self.spec_model.model(
                inputs_embeds=input_embs,
                position_ids=position_ids2,
                attention_mask=tree_masks,
                use_cache=True,
                past_key_values=past_key_values2,
                cache_position=cache_position
            )
            attn_output = outputs["last_hidden_state"]
            if gen_step == 1:
                attn_output = attn_output[:, -1:]
            base_model_output = {"hidden_states": [last_hidden_state[0]]}

        base_model_output["logits"] = self.lm_head(attn_output)
        base_model_output["last_hidden_state"] = last_hidden_state
        base_model_output["past_key_values2"] = past_key_values2

        return base_model_output


    @torch.inference_mode()
    def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            output_logits: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: bool = False,
            streamer: Optional["BaseStreamer"] = None,
            **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        assert len(stopping_criteria) == 2  # max length and EosToken

        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
        last_hidden_state = None
        past_key_values = model_kwargs.get("past_key_values", None)
        past_key_values2 = None
        averaged_accepted_len = []
        gen_step = 0
        tree_masks = None
        tree_tok_scores = []
        tree_tok = []
        parents_list = []
        one_zero = torch.zeros_like(input_ids[0:1, 0:1])
        self.init_tree_mask = self.init_tree_mask.to(input_ids.device)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = dict(past_key_values=past_key_values)

            if past_key_values2 is not None and past_key_values2.get_seq_length() == 0 and gen_step == 1:
                model_inputs["input_ids"] = input_ids[:, -1:]
            elif past_key_values2 is not None and past_key_values2.get_seq_length() > 0 and gen_step == 1:
                model_inputs["input_ids"] = input_ids[:, past_key_values2[0][0].shape[2]:]
            else:
                model_inputs["input_ids"] = input_ids

            if gen_step == self.depth and last_hidden_state is not None:
                # correct cache_position
                if self.tree_decoding:
                    valid_tokens = model_inputs['past_key_values'][0][0].shape[2]

                    scores_list = torch.cat(tree_tok_scores, dim=0).view(-1)
                    ss_token_list = torch.cat(tree_tok, dim=0).view(-1)
                    total_tokens = min(self.total_tokens, scores_list.shape[0])
                    top_scores = torch.topk(scores_list, total_tokens, dim=-1)
                    top_scores_index = top_scores.indices
                    top_scores_index = torch.sort(top_scores_index).values

                    draft_tokens = ss_token_list[top_scores_index]
                    draft_tokens = torch.cat((input_ids[0, valid_tokens:valid_tokens + 1], draft_tokens), dim=0)
                    model_inputs['input_ids'] = draft_tokens[None]

                    draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // self.breadth].long()
                    mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
                    mask_index[draft_parents == 0] = -1
                    mask_index = mask_index + 1

                    tree_mask = torch.full(
                        (total_tokens + 1, total_tokens + 1),
                        fill_value=self.min_dtype,
                        dtype=tree_masks.dtype,
                        device=mask_index.device,
                    )

                    tree_mask2 = torch.eye(total_tokens + 1, device=mask_index.device).bool()
                    tree_mask2[:, 0] = True
                    for i in range(total_tokens):
                        tree_mask2[i + 1].add_(tree_mask2[mask_index[i]])
                    tree_mask[tree_mask2] = 0.0

                    tree_position_ids = torch.sum(tree_mask2, dim=1) - 1
                    model_inputs['position_ids'] = (valid_tokens + tree_position_ids)[None]
                    model_inputs['cache_position'] = model_inputs['position_ids'][0]

                    see_all_valid = torch.zeros((tree_mask.shape[0], valid_tokens), dtype=tree_mask.dtype,
                                                device=tree_mask.device)
                    tree_mask = torch.cat((see_all_valid, tree_mask), -1)
                    model_inputs['attention_mask'] = tree_mask[None, None]
                else:
                    model_inputs['cache_position'] = model_inputs['cache_position'] + torch.arange(
                        model_inputs['input_ids'].shape[1], device=model_inputs['cache_position'].device) - \
                                                     model_inputs['input_ids'].shape[1] + 1

                tree_tok_scores = []
                tree_tok = []
                parents_list = []
            else:
                model_inputs['attention_mask'] = torch.ones_like(model_inputs['input_ids'])
                model_inputs['position_ids'] = model_inputs['attention_mask'].long().cumsum(-1) - 1
                model_inputs['cache_position'] = model_kwargs['cache_position']

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                last_hidden_state=last_hidden_state,
                recall_big=gen_step == self.depth and last_hidden_state is not None,
                past_key_values2=past_key_values2,
                tree_masks=tree_masks,
                gen_step=gen_step
            )

            if gen_step == self.depth or last_hidden_state is None:
                if last_hidden_state is not None and self.verification:
                    nb_layers = len(outputs["past_key_values"])
                    if self.tree_decoding:
                        noleaf_index = torch.unique(mask_index).tolist()
                        noleaf_num = len(noleaf_index) - 1
                        leaf_num = total_tokens - noleaf_num
                        max_depth = torch.max(tree_position_ids) + 1
                        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1

                        position_ids_list = tree_position_ids.tolist()
                        mask_index_list = mask_index.tolist()

                        rid = 0
                        for i in range(total_tokens + 1):
                            if i not in noleaf_index:
                                cid = i
                                depth = position_ids_list[i]
                                for j in reversed(range(depth + 1)):
                                    retrieve_indices[rid][j] = cid
                                    cid = mask_index_list[cid - 1]
                                rid += 1

                        posterior = outputs["logits"][0, retrieve_indices]
                        candidates = torch.cat((model_inputs["input_ids"], one_zero), -1)[0, retrieve_indices]
                        posterior_mask = (
                                candidates[:, 1:] == torch.argmax(posterior[:, :-1], dim=-1)
                        ).int()
                        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
                        accept_length = candidates_accept_length.max()

                        # Choose the best candidate
                        if accept_length == 0:
                            # Default to the first candidate if none are accepted
                            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
                        else:
                            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)

                        input_ids = input_ids[:, :valid_tokens]
                        input_ids = torch.cat(
                            [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)],
                            dim=-1
                        )

                        #  check stopping criterion:
                        for i in range(accept_length.item()):
                            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                                input_ids[:, :valid_tokens + i + 2], None)
                            this_peer_finished = unfinished_sequences.max() == 0
                            if this_peer_finished:
                                input_ids = input_ids[:, :valid_tokens + i + 2]
                                break
                        if this_peer_finished:
                            averaged_accepted_len.append(input_ids.shape[1] - valid_tokens)
                            break

                        averaged_accepted_len.append(accept_length.item() + 1)

                        select_indices = (
                                retrieve_indices[best_candidate, : accept_length + 1] + valid_tokens
                        )
                        for i in range(nb_layers):
                            for k in range(2):
                                layer_cache = outputs['past_key_values'][i]
                                tgt = layer_cache[k][..., select_indices, :]
                                dst = layer_cache[k][..., valid_tokens: valid_tokens + tgt.shape[-2], :]
                                dst.copy_(tgt, non_blocking=True)

                        outputs["logits"] = posterior[best_candidate, accept_length][None, None]
                    else:
                        verifier = model_inputs["input_ids"][0, 1:] == outputs["logits"][0, :-1].argmax(-1)
                        nb_verified_tokens = 1 + torch.cat(
                            (verifier, torch.zeros_like(verifier[0:1]))
                        ).int().argmin().item()
                        tot_verified_tokens = (input_ids.shape[1] - model_inputs["input_ids"].shape[1]
                                               + nb_verified_tokens)

                        # check stopping criterion:
                        for i in range(input_ids.shape[1] - model_inputs["input_ids"].shape[1] + 1,
                                       tot_verified_tokens):
                            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids[:, :i + 1], None)
                            this_peer_finished = unfinished_sequences.max() == 0
                            if this_peer_finished:
                                input_ids = input_ids[:, :i + 1]
                        if this_peer_finished:
                            break

                        input_ids = input_ids[:, :tot_verified_tokens]
                        averaged_accepted_len.append(nb_verified_tokens)

                        outputs["logits"] = outputs["logits"][:, :nb_verified_tokens]

                    outputs["past_key_values"].crop(input_ids.shape[-1])
                    past_key_values2.crop(valid_tokens + 1)
                    outputs["past_key_values2"] = past_key_values2

                    model_kwargs['attention_mask'] = model_kwargs['attention_mask'][:, :input_ids.shape[1]]
                    model_kwargs['cache_position'][-1:] = input_ids.shape[1] - 1

                past_key_values = outputs["past_key_values"]
                gen_step = 0

            last_hidden_state = outputs["hidden_states"]
            past_key_values2 = outputs["past_key_values2"]

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            if self.tree_decoding:
                if gen_step == 0:
                    next_token_logits = outputs["logits"][:, -1, :].log_softmax(-1)
                else:
                    next_token_logits = outputs["logits"].log_softmax(-1)
            else:
                next_token_logits = outputs["logits"][:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            if self.tree_decoding:
                if gen_step == 0:
                    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
                    tree_masks = torch.zeros(
                        size=(1, 1, 1, input_ids.shape[1] + 1),
                        dtype=self.dtype,
                        device=input_ids.device
                    )
                elif gen_step == 1:
                    topk_ind_val = torch.topk(next_tokens_scores.view(-1), self.breadth, dim=-1)
                    next_tokens, topk_cs_p = topk_ind_val.indices, topk_ind_val.values
                    tree_masks = torch.cat((tree_masks.repeat(1, 1, self.breadth, 1), self.init_tree_mask), dim=3)

                    tree_tok_scores.append(topk_cs_p[None])
                    tree_tok.append(next_tokens[None])
                    parents_list.append(torch.zeros(1, dtype=torch.long, device=topk_cs_p.device))
                    topk_cs_index = torch.arange(self.breadth, device=topk_cs_p.device)
                else:
                    ii = (gen_step - 2)
                    bias1 = self.breadth if ii > 0 else 0
                    bias2 = max(0, ii - 1)
                    bias = 1 + self.breadth ** 2 * bias2 + bias1
                    parents = (topk_cs_index + bias)
                    parents_list.append(parents)

                    topk_ind_val2 = torch.topk(next_tokens_scores, self.breadth, dim=-1)
                    next_tokens2, topk_cs_p2 = topk_ind_val2.indices, topk_ind_val2.values

                    cu_scores = topk_cs_p2 + topk_cs_p[:, None]

                    topk_cs = torch.topk(cu_scores.view(-1), self.breadth, dim=-1)
                    topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values

                    out_ids = topk_cs_index // self.breadth
                    next_tokens = next_tokens2.view(-1)[topk_cs_index]

                    tree_masks = torch.cat((tree_masks[:, :, out_ids], self.init_tree_mask), dim=3)

                    tree_tok_scores.append(cu_scores[0])
                    tree_tok.append(next_tokens2[0])
            else:
                next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[None, :]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # if eos_token was found in one sentence, set sentence to finished
            if not self.verification or gen_step == 0:
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
                this_peer_finished = unfinished_sequences.max() == 0

            gen_step += 1

        if streamer is not None:
            streamer.end()

        if self.verification:
            self.stats.extend(averaged_accepted_len)

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
