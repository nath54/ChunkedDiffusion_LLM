#
### Import Modules. ###
#
from typing import Optional, Iterator
#
import torch
from torch import Tensor
from torch import nn
#
from transformers import PreTrainedModel, PreTrainedTokenizer
#
from lib_load_from_hugging_face import load_model, load_tokenizer
from lib_chunked_diffusion_model_config import ChunkedDiffusionModelConfig
from lib_chunks import Chunk
from lib_get_device import get_best_device



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
        self.model_embedding_layer: Optional[nn.Module] = None  # Will be initialized with prepare_model()
        self.model_transformer_layers: Optional[nn.ModuleList] = None  # Will be initialized with prepare_model()
        self.model_lm_head: Optional[nn.Module] = None  # Will be initialized with prepare_model()
        #
        self.projector: nn.Linear = nn.Linear(
            in_features = self.config.from_model_config_hidden_size,
            out_features = (self.config.from_model_config_hidden_size - self.config.permissions_mask_nb_items)
        )
        #
        self.prepare_model()


    #
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:

        #
        return iter( list(self.projector.parameters(recurse=recurse)) + list(self.model.parameters(recurse=recurse)) )


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
    def forward(
        self,
        input_ids: Tensor,  # Dim: (B?, C, d_E)
        permissions_mask: Tensor,  # Dim: (B?, C, k)
        attention_causal_mask: Tensor
    ) -> tuple[Tensor, Tensor]:

        #
        hidden_state: Tensor = self.model_embedding_layer(input_ids)  # type: ignore
        # Dim: (B?, C, d_E)

        #
        for layer in self.model_transformer_layers:  # type: ignore

            #
            projected_hidden_state: Tensor = self.projector( hidden_state )
            # Dim: (B?, C, d_E - k)

            #
            hidden_state = torch.cat( tensors=[projected_hidden_state, permissions_mask], dim=-1 )  # Dim: (B?, C, d_E)

            #
            hidden_state = layer( hidden_state, attention_mask=attention_causal_mask )

        #
        logits: Tensor = self.model_lm_head( hidden_state )  # type: ignore

        #
        return logits, hidden_state


#
### Class for the Full ChunkedDiffusionSystem. ###
#
class ChunkedDiffusionSystem:

    #
    def __init__(
        self,
        model_config: ChunkedDiffusionModelConfig,
        mode: str = "simple",
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = get_best_device()
    ) -> None:

        #
        self.dtype: torch.dtype = dtype
        self.device: str | torch.device = device

        #
        self.model: ChunkedDiffusionModel = ChunkedDiffusionModel(
            config=model_config
        )
        #
        self.tokenizer: PreTrainedTokenizer = load_tokenizer(model_name=model_config.from_model_name, tokenizer_padding_side=model_config.tokenizer_padding_side)

        #
        self.chunks_documents_names: list[ str ] = []
        self.chunks_documents_idx: list[int] = []
        self.chunks: list[ Chunk ] = []
        #
        self.current_chunk: Optional[tuple[int, int]] = None


    #
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:

        #
        return self.model.parameters(recurse=recurse)


    #
    def create_chunk_from_list_of_tokens(self, chunk_tok_ids: list[int], override_chunk_global_lenght: Optional[int] = None) -> Chunk:

        #
        return Chunk(
            permissions_items=self.model.config.permissions_mask_indexes,
            chunk_length = self.model.config.chunk_length,
            chunk_global_context_length = override_chunk_global_lenght if override_chunk_global_lenght is not None else self.model.config.chunk_global_context_length,
            initial_data=torch.tensor(chunk_tok_ids, dtype=torch.int64, device=self.device),
            initial_data_permissions_mask=None,
            padding_token=self.model.config.tokenizer_pad_token,
            dtype=self.dtype,
            device=self.device
        )


    #
    def split_text_of_one_document_in_chunks(self, text: str, override_chunk_global_lenght: Optional[int] = None) -> list[Chunk]:

        #
        text_chunks: list[Chunk] = []

        #
        new_line_tok: list[int] = self.tokenizer.encode("\n")  # type: ignore

        #
        text_sublines: list[str] = []

        #
        text_lines: list[str] = text.split("\n")

        #
        line: str
        #
        for line in text_lines:
            #
            text_sublines += [t + "." for t in line.split(".")]

        #
        current_chunk_token_ids: list[int] = []
        current_chunk_nb_tokens: int = 0

        #
        subline: str
        #
        for subline in text_lines:

            #
            subline_toks: list[int] = self.tokenizer.encode(subline) + new_line_tok  # type: ignore
            #
            subline_nb_toks: int = len(subline_toks)

            #
            ### Case 1: If the current subline fits in the space left of the chunk, add it to it. ###
            #
            if current_chunk_nb_tokens + subline_nb_toks <= self.model.config.chunk_length:
                #
                current_chunk_token_ids += subline_toks
                current_chunk_nb_tokens += subline_nb_toks

            #
            ### Case 2: The current chunk is free, and the current subline is too large, we hard split the subline. ###
            #
            elif current_chunk_nb_tokens == 0:
                #
                ### TODO: maybe improve here by splitting on spaces instead of just filling each chunks to avoid bad words cuts. ###
                #
                while subline_nb_toks >= self.model.config.chunk_length:
                    #
                    text_chunks.append(
                        self.create_chunk_from_list_of_tokens(chunk_tok_ids=subline_toks[:self.model.config.chunk_length], override_chunk_global_lenght=override_chunk_global_lenght)
                    )
                    #
                    subline_toks = subline_toks[self.model.config.chunk_length:]
                    subline_nb_toks = len(subline_toks)
                #
                current_chunk_token_ids += subline_toks
                current_chunk_nb_tokens += subline_nb_toks

            #
            ### Case 3: The current chunk just cannot fit the current subline, so we free the create a new chunk with the current subline. ###
            #
            else:
                #
                text_chunks.append(
                    self.create_chunk_from_list_of_tokens(chunk_tok_ids=current_chunk_token_ids, override_chunk_global_lenght=override_chunk_global_lenght)
                )
                #
                current_chunk_token_ids = subline_toks
                current_chunk_nb_tokens = subline_nb_toks

        #
        return text_chunks


    #
    def split_text_in_chunks(
        self,
        text: str,
        documents: Optional[dict[str, str]] = None,
    ) -> tuple[list[str], list[int], list[Chunk]]:

        #
        documents_chunks: dict[str, list[Chunk]] = {}

        #
        if documents is not None:
            #
            documents_chunks = {
                document_name: self.split_text_of_one_document_in_chunks(text=document_text)
                for document_name, document_text in documents.items()
            }

        #
        text_chunks: list[Chunk] = self.split_text_of_one_document_in_chunks(text=text)

        #
        chunks_documents: list[str] = []
        chunks_documents_idx: list[int] = []
        chunks: list[Chunk] = []

        #
        chunk: Chunk
        #
        for document_idx, document_title in enumerate( documents_chunks ):

            #
            chunks_documents.append( document_title )

            #
            for chunk in documents_chunks[document_title]:
                #
                chunks_documents_idx.append( document_idx )
                #
                chunks.append(chunk)

        #
        main_document_idx: int = len(chunks_documents)
        #
        chunks_documents.append( "main context" )
        #
        for chunk in text_chunks:
            #
            chunks_documents_idx.append( main_document_idx )
            #
            chunks.append(chunk)

        #
        return chunks_documents, chunks_documents_idx, chunks


    #
    def init_all_chunks_global_context_with_chunk_encoding(self) -> Tensor:

        #
        ### TODO: for all chunks, intialize them by encoding them. ###
        #
        pass

        #
        return Tensor()


    #
    def prepare_context_and_masks_for_one_chunk(
        self,
        chunk: Chunk,
        with_globals: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:  # Returns (context tokens, permissions masks, causal mask)

        #
        context_tokens: Tensor
        permissions_mask: Tensor

        #
        ### With globals. ###
        #
        if with_globals:
            #
            ### Concatenating in the sequence dimension, whereas there is a batch_size or not. ###
            #
            context_tokens = torch.cat( tensors=[
                chunk.chunk_context_data,
                chunk.chunk_global_context_data
            ], dim = -1 )
            #
            permissions_mask = torch.cat( tensors=[
                chunk.permission_mask_context_data,
                chunk.permission_mask_global_context_data
            ], dim = -2 )

        #
        ### Without globals. ###
        #
        else:
            #
            context_tokens = chunk.chunk_context_data
            permissions_mask = chunk.permission_mask_context_data

        #
        ### Compute sequence length (works for batched or unbatched). ###
        #
        seq_len: int = permissions_mask.shape[-2]

        #
        ### Create lower triangular matrix (True where can attend causally) ###
        #
        tri: Tensor = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))

        #
        ### Identify non-hidden positions (first permission > 0.5 means hidden). ###
        #
        not_hidden: Tensor = permissions_mask[..., self.model.config.permissions_mask_indexes["hidden"]] <= 0.5  # Shape: (B?, seq_len)

        #
        ### Broadcast not_hidden to mask entire src columns for hidden tokens ###
        ### & combines causal with hidden mask (elementwise AND) ###
        #
        ### Batched case. ###
        #
        if permissions_mask.ndim == 3:
            #
            causal_mask: Tensor = tri[None, :, :] & not_hidden[:, None, :]
            causal_mask = causal_mask.unsqueeze(1)  # Add head dim: (B, 1, seq, seq)
        #
        ### Unbatched case. ###
        #
        else:
            #
            causal_mask: Tensor = tri & not_hidden[None, :]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

        #
        return context_tokens, permissions_mask, causal_mask


    #
    def encode_one_chunk(self, chunk: Chunk) -> Tensor:

        #
        context_tokens, permissions_mask, causal_mask = self.prepare_context_and_masks_for_one_chunk(chunk=chunk, with_globals=True)

        #
        globals_idx: Tensor = ( permissions_mask[ :, self.model.config.permissions_mask_indexes["chunk_global_read_and_write"] ] > 0.5 )

        #
        _logits, hidden_states = self.model.forward(
            input_ids=context_tokens,
            permissions_mask=permissions_mask,
            attention_causal_mask=causal_mask
        )

        #
        return hidden_states[globals_idx]


    #
    def simple_encode_text(self, text: str, encoding_length: Optional[int] = None) -> Tensor:

        #
        chunks: list[Chunk] = self.split_text_of_one_document_in_chunks(text=text, override_chunk_global_lenght=encoding_length)

        #
        chunks_encoding: list[Tensor] = []

        #
        for chunk in chunks:

            #
            chunks_encoding.append(
                self.encode_one_chunk( chunk=chunk )
            )

        #
        if not chunks_encoding:
            #
            raise UserWarning(f"Error: no chunks={chunks} | chunks_encoding={chunks_encoding} for input text=`{text}`")

        #
        final_embedding_tensor: Tensor = chunks_encoding[0]
        #
        for chunk in chunks_encoding[1:]:
            #
            final_embedding_tensor += chunk

        #
        final_embedding_tensor /= float( len(chunks_encoding) )

        #
        return final_embedding_tensor

