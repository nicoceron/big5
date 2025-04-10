import inspect
import warnings

import torch
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import transformers
from transformers.utils import is_torchdynamo_compiling, logging
from transformers.cache_utils import StaticCache
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import SampleOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.generation.streamers import BaseStreamer
from transformers.generation import SampleEncoderDecoderOutput, SampleDecoderOnlyOutput   


logger = logging.get_logger(__name__)

NEED_SETUP_CACHE_CLASSES_MAPPING = {
    "static": StaticCache,
}

"""
Simulate the preprocessing step in transformers.generation.utils.py:GenerationMixin::generate
"""
def expert_generate_helper(
        model: transformers.AutoModelForCausalLM,
        inputs,
        generation_config,
        logits_processor=None, 
        stopping_criteria=None,
        kwargs={}
    ):
    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    model._validate_model_class()
    tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
    generation_config, model_kwargs = model._prepare_generation_config(generation_config, **kwargs)
    model._validate_model_kwargs(model_kwargs.copy())
    
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    accepts_attention_mask = "attention_mask" in set(inspect.signature(model.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]
    
    device = inputs_tensor.device
    model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    # decoder-only models must use left-padding for batched generation.
    if not model.config.is_encoder_decoder and not is_torchdynamo_compiling():
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config.pad_token_id is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )
    
    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache
    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    if model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )
    
    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if model.config.is_encoder_decoder:
        input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = model._prepare_generated_length(
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
        if not model._supports_cache_class:
            raise ValueError(
                "This model does not support the `cache_implementation` argument. Please check the following "
                "issue: https://github.com/huggingface/transformers/issues/28981."
            )
        if generation_config.cache_implementation == "static":
            if not model._supports_static_cache:
                raise ValueError(
                    "This model does not support `cache_implementation='static'`. Please check the following "
                    "issue: https://github.com/huggingface/transformers/issues/28981"
                )
            model_kwargs["past_key_values"] = model._get_static_cache(batch_size, generation_config.max_length)

    model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    if model.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {model.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{model.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 8. prepare distribution pre_processing samplers
    prepared_logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=None,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
    )

    # 9. prepare stopping criteria
    prepared_stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
    )
    
    # 11. prepare logits warper
    prepared_logits_warper = (
        model._get_logits_warper(generation_config) if generation_config.do_sample else None
    )
    
    # 12. expand input_ids with `num_return_sequences` additional sequences per batch
    input_ids, model_kwargs = model._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=generation_config.num_return_sequences,
        is_encoder_decoder=model.config.is_encoder_decoder,
        **model_kwargs,
    )
    
    return {
        "input_ids": input_ids,
        # "logits_processor": prepared_logits_processor,
        # "logits_wrapper": prepared_logits_warper,
        # "stopping_criteria": prepared_stopping_criteria,
        # "generation_config": generation_config,
        # "synced_gpus": None,
        # "streamer": None,
    }, model_kwargs
    
"""
Add three more arguments to model_kwargs:
- expert_logits: Optional[Tensor] = None
- antiexpert_logits: Optional[Tensor] = None
- alpha: Optional[float] = None
"""
def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config.pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
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
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        count_new_tokens = 0
        input_ids_starting_shape = input_ids.shape[-1]
        
        alpha = model_kwargs.get("alpha", 0.0)
        
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            
            ### Start DExpert implementation
            # if the generator model generates less than 5 new tokens, skip this if condition
            input_ids_expert = model_kwargs.get("input_ids_expert", None)
            generate_info_expert = None
            if input_ids_expert is not None:
                input_ids_expert = torch.cat([input_ids_expert, input_ids[:, input_ids_starting_shape:]], dim=-1)
                generate_info_expert = expert_generate_helper(
                    self.expert,
                    input_ids_expert,
                    generation_config=generation_config,
                    logits_processor=None,
                    stopping_criteria=None,
                    kwargs={},
                )
                input_ids_expert = generate_info_expert[0]['input_ids']
                
                # append new input_ids from the generator model
                if count_new_tokens >= 5:
                    # model_kwargs
                    model_inputs_expert = self.expert.prepare_inputs_for_generation(input_ids_expert, **generate_info_expert[1])
                    # expert model forward to get next_token_logits_expert
                    outputs_expert = self.expert(
                        **model_inputs_expert,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                    next_token_logits_expert = outputs_expert.logits[:, -1, :]
                    next_token_logits = next_token_logits + alpha * next_token_logits_expert.to(next_token_logits.device)
            
            
            input_ids_antiexpert = model_kwargs.get("input_ids_antiexpert", None)
            generate_info_antiexpert = None
            if input_ids_antiexpert is not None:
                input_ids_antiexpert = torch.cat([input_ids_antiexpert, input_ids[:, input_ids_starting_shape:]], dim=-1)
                generate_info_antiexpert = expert_generate_helper(
                    self.antiexpert,
                    input_ids_antiexpert,
                    generation_config=generation_config,
                    logits_processor=None,
                    stopping_criteria=None,
                    kwargs={},
                )
                input_ids_antiexpert = generate_info_antiexpert[0]['input_ids']
                
                # append new input_ids from the generator model
                if count_new_tokens >= 5:
                    # model_kwargs
                    model_inputs_antiexpert = self.antiexpert.prepare_inputs_for_generation(input_ids_antiexpert, **generate_info_antiexpert[1])
                    # expert model forward to get next_token_logits_expert
                    outputs_antiexpert = self.antiexpert(
                        **model_inputs_antiexpert,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                    next_token_logits_antiexpert = outputs_antiexpert.logits[:, -1, :]
                    next_token_logits = next_token_logits - alpha * next_token_logits_antiexpert.to(next_token_logits.device)
            
            ### End DExpert implementation

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
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

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if pad_token_id is None:
                pad_token_id = generation_config.eos_token_id[0]
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            
            count_new_tokens += 1

        if streamer is not None:
            streamer.end()

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

def evolve_dexpert_sampling():
    transformers.generation.utils.GenerationMixin._sample = sample