# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import copy
import inspect
import warnings
import torch
from transformers import LogitsProcessorList, MinLengthLogitsProcessor
from transformers import GenerationConfig, StoppingCriteriaList, PreTrainedModel
from transformers.generation import CandidateGenerator
from typing import Union, Optional, Callable, List, Dict
from transformers.generation.candidate_generator import _prepare_attention_mask, _prepare_token_type_ids
from transformers.generation.candidate_generator import _crop_past_key_values
from transformers.generation.utils import NEED_SETUP_CACHE_CLASSES_MAPPING, GenerateNonBeamOutput
from transformers.generation.utils import _split_model_outputs, GenerateDecoderOnlyOutput
from transformers.utils import is_torchdynamo_compiling


class SpecDecBase(torch.nn.Module):

    def __init__(self, big_model, small_model):
        super().__init__()
        self.big_model = big_model
        self.small_model = small_model
        self.device = self.big_model.device

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            **kwargs,
    ) -> tuple[Union[GenerateDecoderOnlyOutput, torch.LongTensor], dict]:

        # Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self.big_model._validate_model_class()
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        generation_config, model_kwargs = self.big_model._prepare_generation_config(generation_config, **kwargs)
        self.big_model._validate_model_kwargs(model_kwargs.copy())

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.big_model.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self.big_model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self.big_model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.big_model.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                    generation_config.pad_token_id is not None
                    and batch_size > 1
                    and len(inputs_tensor.shape) == 2
                    and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                warnings.warn(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.big_model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self.big_model._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # Prepare `input_ids` which will be used for auto-regressive generation
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self.big_model._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        if generation_config.cache_implementation is not None and model_kwargs.get("past_key_values") is not None:
            raise ValueError(
                "Passing both `cache_implementation` (used to initialize certain caches) and `past_key_values` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
            if not self.big_model._supports_cache_class:
                raise ValueError(
                    "This model does not support the `cache_implementation` argument. Please check the following "
                    "issue: https://github.com/huggingface/transformers/issues/28981."
                )
            if generation_config.cache_implementation == "static":
                if not self.big_model._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                model_kwargs["past_key_values"] = self.big_model._get_static_cache(batch_size, generation_config.max_length)

        self.big_model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self.big_model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
        )

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self.big_model._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        if generation_config.num_return_sequences > 1:
            raise ValueError(
                "num_return_sequences has to be 1 when doing assisted generate, "
                f"but is {generation_config.num_return_sequences}."
            )

        if batch_size > 1:
            raise ValueError("assisted generate is only supported for batch_size = 1")
        if not model_kwargs["use_cache"]:
            raise ValueError("assisted generate requires `use_cache=True`")
        if generation_config.cache_implementation == "static":
            raise ValueError("assisted generate is not supported with `static_cache`")

        # 11. Get the candidate generator, given the parameterization
        if generation_config.prompt_lookup_num_tokens is not None:
            candidate_generator = PromptLookupCandidateGenerator(
                num_output_tokens=generation_config.prompt_lookup_num_tokens,
                max_matching_ngram_size=generation_config.max_matching_ngram_size,
                max_length=generation_config.max_length,
            )
        else:
            candidate_generator = SpeculativeCandidateGenerator(
                assistant_model=self.small_model,
                generation_config=generation_config,
                model_kwargs=model_kwargs,
                logits_processor=logits_processor,
            )

        # 12. prepare logits warper (if `do_sample` is `True`)
        prepared_logits_warper = (
            self.big_model._get_logits_warper(generation_config) if generation_config.do_sample else None
        )

        # 13. run assisted/speculative generate
        return self.speculative_decoding(
            input_ids,
            candidate_generator=candidate_generator,
            logits_processor=prepared_logits_processor,
            logits_warper=prepared_logits_warper,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            **model_kwargs,
        )

    def speculative_decoding(
            self,
            input_ids: torch.LongTensor,
            candidate_generator: CandidateGenerator,
            logits_processor: LogitsProcessorList,
            logits_warper: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            **model_kwargs,
    ) -> tuple[Union[GenerateNonBeamOutput, torch.LongTensor], dict]:

        # init values
        do_sample = logits_warper is not None
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        metrics = dict(generated_tokens=[], accepted_tokens=[])

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self.big_model._get_initial_cache_position(input_ids, model_kwargs)

        this_peer_finished = False
        while self.big_model._has_unfinished_sequences(this_peer_finished, False, device=input_ids.device):
            cur_len = input_ids.shape[-1]

            #  1. Fetch candidate sequences from a `CandidateGenerator`
            candidate_input_ids = candidate_generator.get_candidates(input_ids)
            candidate_input_ids = candidate_input_ids.to(self.device)

            candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
            metrics["generated_tokens"].append(candidate_length)

            # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
            # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
            # we use this forward pass to also pick the subsequent logits in the original model.

            # 2.1. Prepare the model inputs
            candidate_kwargs = copy.copy(model_kwargs)
            candidate_kwargs = _prepare_attention_mask(
                candidate_kwargs, candidate_input_ids.shape[1], self.big_model.config.is_encoder_decoder
            )
            candidate_kwargs = _prepare_token_type_ids(candidate_kwargs, candidate_input_ids.shape[1])
            if "cache_position" in candidate_kwargs:
                candidate_kwargs["cache_position"] = torch.cat(
                    (
                        candidate_kwargs["cache_position"],
                        torch.arange(cur_len, cur_len + candidate_length, device=input_ids.device, dtype=torch.long),
                    ),
                    dim=0,
                )

            model_inputs = self.big_model.prepare_inputs_for_generation(candidate_input_ids, **candidate_kwargs)
            if "num_logits_to_keep" in model_inputs:
                model_inputs["num_logits_to_keep"] = candidate_length + 1

            if candidate_length > 0:
                model_inputs['attention_mask'] = model_inputs['attention_mask'].repeat(candidate_input_ids.size(0), 1)
                model_inputs['position_ids'] = model_inputs['position_ids'].repeat(candidate_input_ids.size(0), 1)
            else:
                model_inputs['past_key_values'] = [tuple(layer[i][0, ...].unsqueeze(0) for i in [0, 1]) for layer in model_inputs['past_key_values']]

            # 2.2. Run a forward pass on the candidate sequence
            outputs = self.big_model(
                **model_inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # 2.3. Process the new logits
            new_logits = outputs.logits[:, -candidate_length - 1:]  # excludes the input prompt if present
            next_token_logits = new_logits.clone()
            if len(logits_processor) > 0:
                for i in range(candidate_length + 1):
                    new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])
            if do_sample and len(logits_warper) > 0:
                for i in range(candidate_length + 1):
                    new_logits[:, i, :] = logits_warper(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])

            # 3. Select the accepted tokens.
            if do_sample:
                probs = new_logits.softmax(dim=-1)
                selected_tokens = torch.stack([torch.multinomial(probs[i], num_samples=1).squeeze(1)
                                               for i in range(candidate_input_ids.size(0))])
            else:
                selected_tokens = new_logits.argmax(dim=-1)

            candidate_new_tokens = candidate_input_ids[:, cur_len:]
            matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum(dim=-1)
            n_matches = matches.max()
            match_id = matches.argmax()
            metrics["accepted_tokens"].append(int(n_matches))

            # Ensure we don't generate beyond max_len or an EOS token
            if n_matches > 0:
                is_done_candidate = stopping_criteria(selected_tokens[match_id, : n_matches].unsqueeze(0), None)
                if is_done_candidate:
                    n_matches -= 1
            valid_tokens = selected_tokens[match_id, : n_matches + 1].unsqueeze(0)

            if candidate_length == 0:
                metrics["accepted_tokens"].pop()
                metrics["generated_tokens"].pop()
            # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
            # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
            # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
            # is no match.

            # 4.1. Get the valid continuation, after the matching tokens
            input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
            new_cur_len = input_ids.shape[-1]

            # 4.2. Discard past key values relative to unused assistant tokens
            new_cache_size = new_cur_len - 1
            outputs.past_key_values = [tuple(layer[i][match_id, ...].unsqueeze(0) for i in [0, 1]) for layer in outputs.past_key_values]
            outputs.past_key_values = _crop_past_key_values(self.big_model, outputs.past_key_values, new_cache_size)
            outputs.past_key_values = [tuple(layer[i].repeat(candidate_input_ids.size(0), 1, 1, 1) for i in [0, 1]) for layer in outputs.past_key_values]

            # 5. Update the candidate generation strategy, if needed
            candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)

            # Store scores, attentions and hidden_states when required
            # Assistant: modified to append one tuple element per token, as in the other generation methods.
            if return_dict_in_generate:
                if output_scores:
                    scores += tuple(new_logits[:, i, :] for i in range(n_matches + 1))
                if output_logits:
                    raw_logits += (next_token_logits,)

                if "past_key_values" not in model_kwargs:
                    added_len = new_cur_len
                else:
                    added_len = n_matches + 1

                if output_attentions:
                    decoder_attentions = _split_model_outputs(
                        decoder_attentions,
                        outputs.attentions,
                        cur_len,
                        added_len,
                        is_decoder_attention=True,
                    )
                if output_hidden_states:
                    decoder_hidden_states = _split_model_outputs(
                        decoder_hidden_states, outputs.hidden_states, cur_len, added_len
                    )

            model_kwargs = self.big_model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=False,
                num_new_tokens=n_matches + 1,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            ), metrics
        else:
            return input_ids, metrics

    def device(self):
        return self.device


class SpeculativeCandidateGenerator(CandidateGenerator):

    def __init__(
        self,
        assistant_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        model_kwargs: Dict,
        stopping_criteria: StoppingCriteriaList = None,
        logits_processor: "LogitsProcessorList" = None,
    ):
        # Make sure all data at the same device as assistant model
        device = assistant_model.device

        # Prepare the assistant and the starting number of candidate tokens
        self.assistant_model = assistant_model
        self.num_assistant_tokens = assistant_model.generation_config.num_assistant_tokens
        self.stopping_criteria = stopping_criteria

        # Prepare the kwargs for the assistant model
        assistant_kwargs = {}
        for key, value in model_kwargs.items():  # deepcopy crashes if we attempt to copy encoder outputs with grads
            if key not in ("encoder_outputs", "assistant_encoder_outputs"):
                assistant_kwargs[key] = (
                    value.detach().to(device) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                )

        self.assistant_kwargs = assistant_kwargs

        # Prepare generation-related options.
        self.logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        self.generation_config = copy.deepcopy(generation_config)
        self.generation_config.return_dict_in_generate = True
        self.generation_config.output_scores = True

        # Disable sampling -- this implementation of assisted generation/speculative decoding uses the assistant
        # greedily to maximize matches. Disables sampling-related flags to prevent warnings
        self.generation_config.do_sample = False
        for attr in ("temperature", "top_p", "min_p", "typical_p", "top_k", "epsilon_cutoff", "eta_cutoff"):
            setattr(self.generation_config, attr, None)

        # avoid unnecessary warnings that min_length is larger than max_new_tokens
        # remove the `MinLengthLogitsProcessor` if exists (NOTE: no need to check for `MinNewTokensLogitsProcessor`)
        self.main_model_min_length = self.generation_config.min_length
        self.generation_config.min_length = 0
        self.generation_config.min_new_tokens = None
        for processor in self.logits_processor:
            if type(processor) == MinLengthLogitsProcessor:
                raise ValueError(
                    "Passing `MinLengthLogitsProcessor` when using `assisted_generation is disabled. "
                    "Please pass in `min_length` into `.generate()` instead"
                )

    def get_candidates(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.assistant_model.device)

        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        new_cur_len = input_ids.shape[-1]
        max_new_tokens = min(int(self.num_assistant_tokens), self.generation_config.max_length - new_cur_len - 1)
        min_new_tokens = max(min(max_new_tokens, self.main_model_min_length - new_cur_len), 0)
        if max_new_tokens == 0:
            return input_ids

        # 1. If it is not the first round of candidate generation, prepare the inputs based on the input_ids length
        # (which implicitly contains the number of accepted candidates from the previous round)
        has_past_key_values = self.assistant_kwargs.get("past_key_values", None) is not None
        if has_past_key_values:
            new_cache_size = new_cur_len - 1
            self.assistant_kwargs["past_key_values"] = _crop_past_key_values(
                self.assistant_model, self.assistant_kwargs["past_key_values"], new_cache_size - 1
            )  # the assistant does not have the token after the last match, hence the -1

            self.assistant_kwargs = _prepare_attention_mask(
                self.assistant_kwargs, new_cur_len, self.assistant_model.config.is_encoder_decoder
            )
            self.assistant_kwargs = _prepare_token_type_ids(self.assistant_kwargs, new_cur_len)

        # 2. Forecast next N tokens using the assistant model.
        assistant_generation_kwargs = {
            "input_ids": input_ids,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "generation_config": self.generation_config,
            "logits_processor": self.logits_processor,
            "stopping_criteria": self.stopping_criteria
        }

        assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)

        # 3. Update variables for the next round of candidate generation
        self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values

        # 4. Prepare variables for output
        candidate_ids = assistant_output.sequences
        return candidate_ids

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        # do nothing for now
        return


class PromptLookupCandidateGenerator(CandidateGenerator):
    """
    `CandidateGenerator` class to be used for prompt lookup generation. This class generates candidates by looking up
    likely continuations in the provided prompt (input_ids) itself.
    Read the following blog post for more information: https://github.com/apoorvumang/prompt-lookup-decoding

    Args:
        max_matching_ngram_size (`int`):
            The maximum ngram size to be considered for matching in the prompt
        num_output_tokens (`int`):
            The number of tokens to be output as candidate tokens.
        max_length (`int`):
            The number of total maximum tokens that can be generated. For decoder-only models that includes the prompt length.
            Defaults to 20, which is the max length used as default in generation config.
    """

    def __init__(
        self,
        num_output_tokens: int = 10,
        max_matching_ngram_size: int = None,
        max_length: int = 20,
    ):
        self.num_output_tokens = num_output_tokens
        self.max_matching_ngram_size = max_matching_ngram_size if max_matching_ngram_size else 2
        self.max_length = max_length

        if self.max_matching_ngram_size <= 0 or self.num_output_tokens <= 0:
            raise ValueError("Invalid max_matching_ngram_size or num_output_tokens")

    def get_candidates(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(num_candidates, candidate_length)`: The candidate sequences to be tried.
        """
        input_length = input_ids.size(1)

        # Don't generate more than `max_length - 1` candidates since the target model generates one extra token.
        if self.max_length == input_length + 1:
            return input_ids

        chosen_ids = None
        match_found = False
        for ngram_size in range(min(self.max_matching_ngram_size, input_length - 1), 0, -1):
            # Create sliding windows of size ngram_size
            windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

            # Convert ngram to a tensor for comparison
            ngram_tensor = input_ids[0, -ngram_size:]

            # Find where the windows match the ngram
            matches = (windows == ngram_tensor).all(dim=2)

            # Get the indices of matches
            match_indices = matches.nonzero(as_tuple=True)[1]

            # Iterate through match indices to find a valid continuation
            for idx in match_indices:
                start_idx = idx + ngram_size
                end_idx = start_idx + self.num_output_tokens
                end_idx = min(end_idx, input_length, self.max_length)

                if start_idx < end_idx:
                    chosen_ids = input_ids[0, start_idx:end_idx]
                    match_found = True
                    break
            if match_found:
                break

        if chosen_ids is None or len(chosen_ids) == 0:
            # In case we didn't find a match return the input sequence unchanged, reverts back to autoregressive decoding
            return input_ids

        # Now need extend input_ids with chosen_ids
        chosen_ids = chosen_ids.unsqueeze(0)
        candidate_input_ids = torch.cat((input_ids, chosen_ids), dim=1)
        # assisted_generation expects logits as well, but we don't have those here, so returning None
        return candidate_input_ids

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        # Currently does nothing
        return
