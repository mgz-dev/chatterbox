from typing import Optional, Union, Sequence

import torch
from torch import nn as nn
from transformers import LlamaConfig, LlamaPreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from ..llama_model import ModifiedLlamaModel


class T3HuggingfaceBackend(LlamaPreTrainedModel, GenerationMixin):
    """
    Override some HuggingFace interface methods so we can use the standard `generate` method with our
    custom embedding / logit layers.

    NOTE: need to extend "*PreTrainedModel" to avoid re-initializing weights!
    """

    def __init__(
        self,
        config: LlamaConfig,
        llama: ModifiedLlamaModel,
        *,
        speech_enc,
        speech_head,
        latents_queue: Optional[torch.Tensor] = None,
        logits_queue: Optional[torch.Tensor] = None,
        alignment_layer_idx: Optional[Union[int, Sequence[int]]] = None,
    ):
        super().__init__(config)
        self.model = llama
        self.speech_enc = speech_enc
        self.speech_head = speech_head
        self._added_cond = False
        self.alignment_layer_idx = alignment_layer_idx


    @torch.inference_mode()
    def prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, decoder_cond: torch.Tensor, use_cache: bool, past_key_values=None,
        # This argument was introduced in some recent version of transformers (>=4.29.1)
        cache_position=None
    ):
        """
        This is a method used by huggingface's generate() method.
        Overridden here to apply our custom speech token embedding layer.

        :param input_ids: (B, S) int64 tensors of input tokens.
        :param decoder_cond: (B, T, C) float32 tensor of conditioning (prefixed to <input_embeds>)
        """

        # Make use of the kv cache: only the last input ID is new, we trim away all the ones before
        if not use_cache:
            past_key_values = None
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # custom speech token embedding layer
        inputs_embeds = self.speech_enc(input_ids)

        # prefix decoder conditioning if applicable
        if not self._added_cond:
            assert past_key_values is not None # should be first step
            if decoder_cond.size(0) != inputs_embeds.size(0):
                decoder_cond = decoder_cond.expand(inputs_embeds.size(0), -1, -1)
            inputs_embeds = torch.cat([decoder_cond, inputs_embeds], dim=1)
            self._added_cond = True

        return {
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @torch.inference_mode()
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[torch.Tensor]=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        """
        This is a method used by huggingface's generate() method.
        Overridden here to apply our custom layer norm and speech logit projection layers.

        :param inputs_embeds: (B, S, C) float32 tensor of conditioning inputs. If past key values are given,
        S should be 1.
        """
        is_large_input = inputs_embeds.size(1) != 1
        has_cache = past_key_values is not None and len(past_key_values) > 0
        assert not (is_large_input and has_cache)
        assert return_dict
        # assert output_hidden_states

        # We want attentions from this specific layer to power the analyzer
        if self.alignment_layer_idx is None:
            output_attentions_layer_indices = None
        elif isinstance(self.alignment_layer_idx, (list, tuple)):
            output_attentions_layer_indices = list(self.alignment_layer_idx)
        else:
            output_attentions_layer_indices = [self.alignment_layer_idx]

        tfmr_out = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_attentions_layer_indices=output_attentions_layer_indices,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        # hidden_states = tfmr_out.hidden_states[-1]  # (B, seq, dim)
        hidden_states = tfmr_out.last_hidden_state
        logits = self.speech_head(hidden_states)
        # assert inputs_embeds.size(0) == 1 # (disabled for CFG)

        return CausalLMOutputWithCrossAttentions(
            logits=logits,
            past_key_values=tfmr_out.past_key_values,
            hidden_states=tfmr_out.hidden_states,
            attentions=tfmr_out.attentions,
        )
