#
### Import Modules. ###
#
from typing import Optional
#
from torch import Tensor
from torch import nn
#
from transformers import PreTrainedModel
#
from lib_load_from_hugging_face import load_model
#
from lib_chunked_diffusion_model_config import ChunkedDiffusionModelConfig


#
### Class ChunkedDiffusionModel. ###
#
class ChunkedDiffusionModel(nn.Module):

    #
    def __init__(self, config: ChunkedDiffusionModelConfig) -> None:

        #
        self.config: ChunkedDiffusionModelConfig = config
        #
        self.model: PreTrainedModel = load_model(model_name=self.config.from_model_name)
        #
        self.model_embedding_layer: Optional[nn.Module] = None  # TODO
        self.model_transformer_layers: Optional[nn.ModuleList] = None  # TODO
        self.model_lm_head: Optional[nn.Module] = None  # TODO
        #
        self.projector: nn.Linear = nn.Linear(
            in_features = self.config.from_model_config_hidden_size,
            out_features = (self.config.from_model_config_hidden_size - self.config.mask_nb_items)
        )


    #
    def prepare_model(self) -> None:

        """
        Step 1: Disassembles the pre-trained model to isolate its key components:
        the embedding layer, the transformer layers, and the language model head.
        This is necessary to later manipulate or pass data through specific parts of the model.
        """

        #
        ### Prepare `self.model_embedding_layer`, `self.model_transformer_layers`, `self.model_lm_head`. ###
        #
        model_family: str = self.config.from_model_family.lower()

        #
        ### Handle models with a common architecture (e.g., Llama, Qwen, Mistral) ###
        ### These models typically have a `model` attribute containing `embed_tokens` and `layers`. ###
        #
        if "qwen" in model_family or "llama" in model_family or "mistral" in model_family:
            #
            self.model_embedding_layer = self.model.model.embed_tokens  # type: ignore
            self.model_transformer_layers = self.model.model.layers  # type: ignore
            self.model_lm_head = self.model.lm_head  # type: ignore

        #
        ### GPT-2 model family. ###
        #
        elif "gpt2" in model_family:
            #
            self.model_embedding_layer = self.model.transformer.wte  # type: ignore
            self.model_transformer_layers = self.model.transformer.h  # type: ignore
            self.model_lm_head = self.model.lm_head  # type: ignore

        #
        ### Raise an error if the model family is not supported yet. ###
        #
        else:
            #
            raise NotImplementedError(
                f"Model family '{self.config.from_model_family}' is not yet supported. "
                f"Please add its specific layer architecture to the `prepare_model` method."
            )


    #
    def run(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[tuple[tuple[Tensor, Tensor], ...]] = None,
        use_cache: Optional[bool] = None
    ) -> tuple[Tensor, Optional[tuple[tuple[Tensor, Tensor], ...]]]:

        """
        Performs the forward pass of the model using its decomposed components.

        This method supports caching of key-value states for efficient, auto-regressive
        text generation.

        Args:
            input_ids (Tensor): The input token IDs. Shape: (batch_size, sequence_length).
            attention_mask (Optional[Tensor]): Mask to avoid performing attention on padding
                                               token indices. Shape: (batch_size, sequence_length).
            past_key_values (Optional[Tuple[Tuple[Tensor, Tensor], ...]]):
                A tuple containing pre-computed key-value states for each transformer layer.
                Used to speed up sequential decoding.
            use_cache (Optional[bool]): If True, the model returns the updated key-value states.

        Returns:
            A tuple containing:
            - logits (Tensor): The raw, unnormalized scores for each token in the vocabulary.
                               Shape: (batch_size, sequence_length, vocab_size).
            - new_past_key_values (Optional[Tuple[...]]): The updated cache if `use_cache` is True.
        """

        #
        ### Ensure the model components have been initialized from the pre-trained model. ###
        #
        if self.model_embedding_layer is None or self.model_transformer_layers is None or self.model_lm_head is None:
            #
            raise RuntimeError(
                "Model components are not initialized. Please call `prepare_model()` before running the model."
            )

        #
        ### 1. Pass input tokens through the embedding layer. ###
        #
        hidden_states: Tensor = self.model_embedding_layer(input_ids)

        #
        ### 2. Sequentially pass the hidden states through each transformer layer. ###
        #
        present_key_values: Optional[list[tuple[Tensor, Tensor]]] = [] if use_cache else None

        #
        for i, layer in enumerate(self.model_transformer_layers):

            #
            ### Extract the past key-value state for the current layer, if available. ###
            #
            layer_past: Optional[tuple[Tensor, Tensor]] = past_key_values[i] if past_key_values is not None else None

            #
            ### A standard Hugging Face transformer layer returns a tuple. ###
            ### The first element is the new hidden state. ###
            ### The second, if use_cache is True, is the updated key-value pair (the "present" state). ###
            #
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache
            )

            #
            hidden_states = layer_outputs[0]

            #
            if use_cache and present_key_values is not None:
                #
                present_key_values.append(layer_outputs[1])

        #
        ### 3. Pass the final hidden states through the language model head to get logits. ###
        #
        logits = self.model_lm_head(hidden_states)

        #
        ### 4. Assemble the new cache from all layers. ###
        #
        new_past_key_values = tuple(present_key_values) if use_cache and present_key_values is not None else None

        #
        return logits, new_past_key_values
