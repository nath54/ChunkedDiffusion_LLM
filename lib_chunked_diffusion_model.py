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
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, Qwen2DecoderLayer
#
from lib_load_from_hugging_face import load_model, load_tokenizer
from lib_chunked_diffusion_model_config import ChunkedDiffusionModelConfig
from lib_chunks import Chunk, create_permission_vector
from lib_get_device import get_best_device


#
def applst(lst1: list[Tensor], lst2: list[Tensor], itm1: Tensor, itm2: Tensor) -> None:
    #
    lst1.append(itm1)
    lst2.append(itm2)


#
def addlst(lst1: list[Tensor], lst2: list[Tensor], itms1: list[Tensor], itms2: list[Tensor]) -> None:
    #
    lst1 += itms1
    lst2 += itms2


#
### Custom Shared LMHead
#

class SharedEmbeddingAndLMHead(nn.Module):
    """
    A module that shares a single weight matrix for an nn.Embedding layer
    and a linear layer (LM head).
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: Optional[torch.dtype] = None, device: Optional[str | torch.device] = None) -> None:

        #
        super().__init__()  # type: ignore

        #
        ### Use a single nn.Parameter for the weight matrix. ###
        #
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        #
        nn.init.normal_(self.weight)  # Initialize the weight matrix.

        #
        self.device: Optional[str | torch.device] = device
        self.dtype: Optional[torch.dtype] = dtype

    #
    def forward(self, input_ids: Tensor) -> Tensor:
        #
        ### The forward method will be implemented in the main model. ###
        #
        raise NotImplementedError

    #
    def get_embedding_layer(self):
        #
        ### Create an nn.Embedding layer that uses the shared weight matrix. ###
        #
        embedding_layer = nn.Embedding(
            num_embeddings=self.weight.size(0),
            embedding_dim=self.weight.size(1),
            _weight=self.weight,
            device=self.device,
            dtype=self.dtype
        )
        #
        return embedding_layer

    #
    def get_lm_head(self):
        #
        ### Create a linear layer (LM head) that uses the transposed shared weight matrix. ###
        #
        lm_head = nn.Linear(
            in_features=self.weight.size(1),
            out_features=self.weight.size(0),
            bias=False,
            device=self.device,
            dtype=self.dtype
        )
        #
        lm_head.weight = self.weight  # This is the key line for sharing the weights!
        #
        return lm_head


#
### Class ChunkedDiffusionModel. ###
#
class ChunkedDiffusionModel(nn.Module):

    #
    def __init__(
        self,
        config: ChunkedDiffusionModelConfig,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = get_best_device(),
    ) -> None:

        #
        super().__init__()  # type: ignore

        #
        self.dtype: torch.dtype = dtype
        self.device: str | torch.device = device
        #
        self.config: ChunkedDiffusionModelConfig = config
        #
        self.model: Optional[PreTrainedModel] = None
        #
        if self.config.from_model_custom_config is None:
            #
            self.model = load_model(model_name=self.config.from_model_name).to(device=self.device, dtype=self.dtype)  # type: ignore
        #
        self.model_embedding_layer: Optional[nn.Module] = None  # Will be initialized with prepare_model()
        self.model_transformer_layers: Optional[nn.ModuleList] = None  # Will be initialized with prepare_model()
        self.model_lm_head: Optional[nn.Module] = None  # Will be initialized with prepare_model()
        #
        self.model_rotary_embedding: Qwen2RotaryEmbedding = Qwen2RotaryEmbedding(config=self.config.from_model_config, device=self.device)  # type: ignore
        #
        self.projector_intern: nn.Linear = nn.Linear(
            in_features = self.config.from_model_config_hidden_size,
            out_features = (self.config.from_model_config_hidden_size - self.config.permissions_mask_nb_items)
        )

        #
        self.prepare_model()


    #
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:

        #
        params_lst: list[nn.Parameter] = list(self.projector_intern.parameters(recurse=recurse)) + list(self.model_rotary_embedding.parameters())

        #
        if self.model is not None:
            #
            params_lst += list(self.model.parameters(recurse=recurse))
        #
        elif self.model_embedding_layer is not None and self.model_lm_head is not None and self.model_transformer_layers is not None:
            #
            params_lst += list(self.model_embedding_layer.parameters(recurse=recurse))
            params_lst += list(self.model_lm_head.parameters(recurse=recurse))
            params_lst += list(self.model_transformer_layers.parameters(recurse=recurse))

        #
        return iter( params_lst )


    #
    def prepare_model(self) -> None:

        """
        Step 1: Disassembles the pre-trained model to isolate its key components:
        the embedding layer, the transformer layers, and the language model head.
        This is necessary to later manipulate or pass data through specific parts of the model.
        """

        #
        ### If not a custom model config. ###
        #
        if self.config.from_model_custom_config is None:

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
        ### If it is a custom model config. ###
        #
        else:

            #
            if hasattr(self.config.from_model_config, "shared_head_and_embeddings") and getattr(self.config.from_model_config, "shared_head_and_embeddings"):
                #
                shared_embedding_and_head_module: SharedEmbeddingAndLMHead = SharedEmbeddingAndLMHead(
                    num_embeddings=self.config.from_model_config.vocab_size,
                    embedding_dim=self.config.from_model_config.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                #
                self.model_embedding_layer = shared_embedding_and_head_module.get_embedding_layer()
                self.model_lm_head = shared_embedding_and_head_module.get_lm_head()
            #
            else:
                #
                self.model_embedding_layer = nn.Embedding(
                    num_embeddings=self.config.from_model_config.vocab_size,
                    embedding_dim=self.config.from_model_config.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                #
                self.model_lm_head = nn.Linear(
                    in_features=self.config.from_model_config.hidden_size,
                    out_features=self.config.from_model_config.vocab_size,
                    dtype=self.dtype,
                    device=self.device
                )

            #
            self.model_transformer_layers = nn.ModuleList(
                modules = [
                    Qwen2DecoderLayer(config=self.config.from_model_config, layer_idx=i)  # type:ignore
                    for i in range(self.config.from_model_config.num_hidden_layers)
                ]
            ).to(
                dtype=self.dtype,
                device=self.device
            )

            #
            print(f"Custom model config initialized : {self.config.from_model_custom_config}")


    #
    def forward_from_hidden_state(
        self,
        hidden_state: Tensor,   # Dim: (B?, C, d_E)
        permissions_mask: Tensor,  # Dim: (B?, C, k)
        attention_causal_mask: Tensor,
        position_ids: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor]:

        #
        ### Add batch dim if missing (makes inputs consistently 3D/4D). ###
        #
        if hidden_state.ndim == 2:
            #
            hidden_state = hidden_state.unsqueeze(0)
        #
        if permissions_mask.ndim == 2:
            #
            permissions_mask = permissions_mask.unsqueeze(0)

        #
        for layer in self.model_transformer_layers:  # type: ignore

            #
            projected_hidden_state: Tensor = self.projector_intern( hidden_state )
            # Dim: (B?, C, d_E - k)

            #
            hidden_state = torch.cat( tensors=[projected_hidden_state, permissions_mask], dim=-1 )  # Dim: (B?, C, d_E)

            #
            ### Pass position_ids to layer; also fix mask polarity/dtype for SDPA. ###
            ### True now = masked (cannot attend) ###
            #
            # attn_mask = (~attention_causal_mask).to(dtype=torch.bool)
            attn_mask: Tensor = (1.0 - attention_causal_mask.to(self.dtype)) * torch.finfo(self.dtype).min  # type: ignore

            #
            ### [0] because layer returns tuple (hidden, ...) ###
            #
            hidden_state = layer(
                hidden_state,
                attention_mask=attn_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=use_cache
            )[0]

            #
            ### To fix bug that removes the batch dimension. ###
            #
            if hidden_state.ndim == 2:
                #
                hidden_state = hidden_state.unsqueeze(0)

        #
        logits: Tensor = self.model_lm_head( hidden_state )  # type: ignore

        #
        return logits, hidden_state


    #
    def forward(
        self,
        input_ids: Tensor,  # Dim: (B?, C, d_E)
        permissions_mask: Tensor,  # Dim: (B?, C, k)
        attention_causal_mask: Tensor,
        use_cache: bool = False,
        modified_hidden_states: Optional[dict[int, Tensor]] = None,
    ) -> tuple[Tensor, Tensor]:

        #
        ### Add batch dim if missing (makes inputs consistently 3D/4D). ###
        #
        if input_ids.ndim == 1:
            #
            input_ids = input_ids.unsqueeze(0)
        #
        if permissions_mask.ndim == 2:
            #
            permissions_mask = permissions_mask.unsqueeze(0)

        #
        hidden_state: Tensor = self.model_embedding_layer(input_ids)  # type: ignore
        # Dim: (B?, C, d_E)

        #
        if modified_hidden_states is not None:

            #
            for tok_id, modified_hidden_state in modified_hidden_states.items():

                #
                if hidden_state.ndim == 2:
                    #
                    hidden_state[tok_id, :] = modified_hidden_state

                #
                elif hidden_state.ndim == 3:
                    #
                    hidden_state[0, tok_id, :] = modified_hidden_state

        #
        ### Create position_ids: standard 0 to seq-1, expanded for batch ###
        ### Use shape[1] for seq_len (batch-safe) ###
        #
        seq_len: int= hidden_state.shape[1]
        #
        position_ids: Tensor = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(hidden_state.shape[0], seq_len)
        #
        position_embeddings: tuple[Tensor, Tensor] = self.model_rotary_embedding(x=hidden_state, position_ids=position_ids)

        #
        ### Calculate all the layers stack and return the logits and hidden state. ###
        #
        return self.forward_from_hidden_state(
            hidden_state=hidden_state,
            permissions_mask=permissions_mask,
            attention_causal_mask=attention_causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=use_cache
        )


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
        ).to(device=self.device)
        #
        self.tokenizer: PreTrainedTokenizer = load_tokenizer(model_name=model_config.from_model_name, tokenizer_padding_side=model_config.tokenizer_padding_side)

        #
        self.chunks_documents_names: list[ str ] = []
        self.chunks_documents_idx: list[int] = []
        self.chunks: list[ Chunk ] = []
        #
        self.current_chunk: Optional[tuple[int, int]] = None

        #
        """
        "hidden": 0,
        "separation": 1,
        "system_prompt_read_only": 2,
        "file_name_read_only": 3,
        "document_read_only": 4,
        "chunk_inside_read_only": 5,
        "chunk_global_read_only": 6,
        "chunk_inside_read_and_write": 7,
        "chunk_global_read_and_write": 8,
        "global_space_read_and_write": 9,
        "next_token_prediction_cursor": 10,
        """
        #
        self.permissions_vectors: dict[str, Tensor] = {

            permission_name: create_permission_vector(
                nb_permissions_items=self.model.config.permissions_mask_nb_items,
                permission_item = item_idx,
                dtype=self.dtype,
                device=self.device
            )

            for permission_name, item_idx in self.model.config.permissions_mask_indexes.items()
        }
        #
        self.toks: dict[str, Tensor] = {
            "pad": torch.tensor(data=[self.model.config.tokenizer_documents_separation_token], dtype=torch.int64, device=self.device),
            "chunk_sep": torch.tensor(data=[self.model.config.tokenizer_documents_separation_token], dtype=torch.int64, device=self.device),
            "doc_sep": torch.tensor(data=[self.model.config.tokenizer_documents_separation_token], dtype=torch.int64, device=self.device),
            "doc_title_content_sep": torch.tensor(data=[self.model.config.tokenizer_documents_separation_token], dtype=torch.int64, device=self.device),
        }


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
            initial_data=torch.tensor(chunk_tok_ids, dtype=torch.int64, device=self.device) if chunk_tok_ids else None,
            initial_data_permissions_mask=None,
            padding_token=self.model.config.tokenizer_pad_token,
            dtype=self.dtype,
            device=self.device
        )


    #
    def split_text_of_one_document_in_chunks(self, text: str, override_chunk_global_lenght: Optional[int] = None) -> tuple[ list[Chunk], list[int] ]:

        #
        text_chunks: list[Chunk] = []
        chunks_lengths: list[int] = []

        #
        text_sublines: list[str] = []

        #
        text_lines: list[str] = text.split("\n")

        #
        line: str
        #
        for line in text_lines:
            #
            text_sublines += [t + "." for t in (line + "\n").split(".")]

        #
        current_chunk_token_ids: list[int] = []
        current_chunk_nb_tokens: int = 0

        #
        subline: str
        #
        for subline in text_sublines:

            #
            subline_toks: list[int] = self.tokenizer.encode(subline)  # type: ignore
            #
            subline_toks = [t % self.model.config.from_model_config.vocab_size for t in subline_toks]
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
                    chunks_lengths.append( current_chunk_nb_tokens )
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
                chunks_lengths.append( current_chunk_nb_tokens )
                #
                current_chunk_token_ids = subline_toks
                current_chunk_nb_tokens = subline_nb_toks

        #
        ### Last chunk if not empty. ###
        #
        if current_chunk_token_ids:
            #
            text_chunks.append(
                self.create_chunk_from_list_of_tokens(chunk_tok_ids=current_chunk_token_ids, override_chunk_global_lenght=override_chunk_global_lenght)
            )
            #
            chunks_lengths.append( current_chunk_nb_tokens )

        #
        return text_chunks, chunks_lengths


    #
    def split_text_in_chunks(
        self,
        text: str,
        documents: Optional[dict[str, str]] = None,
    ) -> tuple[list[str], list[int], list[Chunk], list[int]]:

        #
        documents_chunks: dict[str, tuple[list[Chunk], list[int]]] = {}

        #
        if documents is not None:
            #
            documents_chunks = {
                document_name: self.split_text_of_one_document_in_chunks(text=document_text)
                for document_name, document_text in documents.items()
            }

        #
        text_chunks: list[Chunk]
        text_chunks_length: list[int]
        #
        text_chunks, text_chunks_length = self.split_text_of_one_document_in_chunks(text=text)

        #
        chunks_documents: list[str] = []
        chunks_documents_idx: list[int] = []
        chunks: list[Chunk] = []
        chunks_lengths: list[int] = []

        #
        chunk: Chunk
        #
        for document_idx, document_title in enumerate( documents_chunks ):

            #
            chunks_documents.append( document_title )

            #
            for chunk, chunk_length in zip(*documents_chunks[document_title]):
                #
                chunks_documents_idx.append( document_idx )
                #
                chunks.append(chunk)
                chunks_lengths.append(chunk_length)

        #
        main_document_idx: int = len(chunks_documents)
        #
        chunks_documents.append( "main context" )
        #
        for chunk, chunk_length in zip( text_chunks, text_chunks_length ):
            #
            chunks_documents_idx.append( main_document_idx )
            #
            chunks.append(chunk)
            chunks_lengths.append(chunk_length)

        #
        return chunks_documents, chunks_documents_idx, chunks, chunks_lengths


    #
    def prepare_attention_causal_mask_from_permissions_mask(
        self,
        permissions_mask: Tensor,
        hide_hidden: bool = False,
    ) -> Tensor:

        #
        ### Compute sequence length (works for batched or unbatched). ###
        #
        seq_len: int = permissions_mask.shape[-2]

        #
        ### Create lower triangular matrix (True where can attend causally) ###
        #
        tri: Tensor = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))

        #
        attention_causal_mask: Tensor
        #
        ### Hide hiddens. ###
        #
        if hide_hidden:

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
                attention_causal_mask = tri[None, :, :] & not_hidden[:, None, :]
                attention_causal_mask = attention_causal_mask.unsqueeze(1)  # Add head dim: (B, 1, seq, seq)
            #
            ### Unbatched case. ###
            #
            else:
                #
                attention_causal_mask = tri & not_hidden[None, :]
                attention_causal_mask = attention_causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

        #
        else:

            #
            ### Broadcast not_hidden to mask entire src columns for hidden tokens ###
            #
            ### Batched case. ###
            #
            if permissions_mask.ndim == 3:
                #
                attention_causal_mask = tri[None, :, :]
                attention_causal_mask = attention_causal_mask.unsqueeze(1)  # Add head dim: (B, 1, seq, seq)
            #
            ### Unbatched case. ###
            #
            else:
                #
                attention_causal_mask = tri
                attention_causal_mask = attention_causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

        #
        return attention_causal_mask


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
        attention_causal_mask: Tensor = self.prepare_attention_causal_mask_from_permissions_mask(permissions_mask=permissions_mask)

        #
        return context_tokens, permissions_mask, attention_causal_mask


    #
    def context_and_permissions_from_doc_title(
        self,
        doc_title: str,
    ) -> tuple[Tensor, Tensor]:

        #
        title_toks: list[int] = self.tokenizer.encode(text=doc_title)  # type: ignore
        #
        context: Tensor = torch.tensor(data=title_toks, dtype=torch.int64, device=self.device)
        #
        permissions: Tensor = torch.tile(
            input=self.permissions_vectors["file_name_read_only"],
            dims=(len(title_toks), 1)
        )

        #
        return (context, permissions)


    #
    def encode_one_chunk(self, chunk: Chunk, chunk_modified_hidden_states: Optional[dict[int, Tensor]] = None) -> Tensor:

        #
        context_tokens, permissions_mask, causal_mask = self.prepare_context_and_masks_for_one_chunk(chunk=chunk, with_globals=True)

        #
        if permissions_mask.ndim == 2:
            #
            globals_idx: Tensor = ( permissions_mask[ :, self.model.config.permissions_mask_indexes["chunk_global_read_and_write"] ] > 0.5 )
        #
        elif permissions_mask.ndim == 3:
            #
            ### Now batch-safe (B, seq). ###
            #
            globals_idx = (permissions_mask[:, :, self.model.config.permissions_mask_indexes["chunk_global_read_and_write"]] > 0.5)
        #
        else:
            #
            raise UserWarning(f"Bad permissions_mask ndim = {permissions_mask.ndim}")

        #
        _logits, hidden_states = self.model.forward(
            input_ids=context_tokens,
            permissions_mask=permissions_mask,
            attention_causal_mask=causal_mask,
            modified_hidden_states=chunk_modified_hidden_states
        )

        #
        if globals_idx.ndim == 1:
            #
            globals_idx = globals_idx.unsqueeze(0)

        #
        return hidden_states[globals_idx]


    #
    def simple_encode_text(self, text: str, encoding_length: Optional[int] = None) -> Tensor:

        #
        chunks: list[Chunk]
        _chunks_lengths: list[int]
        #
        chunks, _chunks_lengths = self.split_text_of_one_document_in_chunks(text=text, override_chunk_global_lenght=encoding_length)

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


    #
    def init_all_chunks_global_context_with_chunk_encoding(
        self,
        chunks: list[Chunk]
    ) -> list[Chunk]:

        #
        for chunk in chunks:

            #
            chunk.chunk_global_context_data = self.encode_one_chunk(chunk=chunk)

        #
        return chunks


    #
    def prepare_context_and_masks_for_all_chunks(
        self,
        chunks_documents: list[str],
        chunks_documents_idx: list[int],
        chunks: list[Chunk],
        current_chunk: int,
        chunks_modified_hidden_states: Optional[dict[int, dict[int, Tensor]]] = None,
    ) -> tuple[Tensor, Tensor, Tensor, int]:  # Returns (context tokens, permissions masks, causal mask, current_chunk_context_start_pos_idx)

        #
        contexts_tensors: list[ Tensor ] = []
        permissions_tensors: list[ Tensor ] = []
        #
        current_chunk_context_start_pos_idx: int = 0

        #
        ### First document separation and title
        #
        ## Separation. ##
        #
        applst(
            contexts_tensors, permissions_tensors,
            self.toks["doc_sep"], self.permissions_vectors["separation"]
        )
        #
        ## Title. ##
        #
        applst(
            contexts_tensors, permissions_tensors,
            *self.context_and_permissions_from_doc_title(doc_title=chunks_documents[0])
        )
        #
        ## Content and title Separation. ##
        #
        applst(
            contexts_tensors, permissions_tensors,
            self.toks["doc_sep"], self.permissions_vectors["separation"]
        )

        #
        crt_doc_idx: int = 0

        #
        for id_chunk in range(len(chunks)):

            #
            ### If document change. ###
            #
            if chunks_documents_idx[id_chunk] != crt_doc_idx:

                #
                crt_doc_idx = chunks_documents_idx[id_chunk]

                #
                ## Separation. ##
                #
                applst(
                    contexts_tensors, permissions_tensors,
                    self.toks["doc_sep"], self.permissions_vectors["separation"]
                )
                #
                ## Title. ##
                #
                applst(
                    contexts_tensors, permissions_tensors,
                    *self.context_and_permissions_from_doc_title(doc_title=chunks_documents[crt_doc_idx])
                )
                #
                ## Content and title Separation. ##
                #
                applst(
                    contexts_tensors, permissions_tensors,
                    self.toks["doc_sep"], self.permissions_vectors["separation"]
                )

            #
            ### If current chunk. ###
            #
            if current_chunk == id_chunk:

                #
                ### Get the current chunk context pos idx. ###
                #
                current_chunk_context_start_pos_idx = int( sum( [ctx_tens.shape[-2] for ctx_tens in contexts_tensors] ) )

                #
                ### Chunk Context with chunk separation at the end. ###
                #
                applst(
                    contexts_tensors, permissions_tensors,
                    chunks[id_chunk].chunk_context_data, chunks[id_chunk].permission_mask_context_data
                )
                #
                applst(
                    contexts_tensors, permissions_tensors,
                    self.toks["chunk_sep"], self.permissions_vectors["separation"]
                )

            #
            ### If NOT current chunk. ###
            #
            else:
                #
                ### Chunk Global Context with chunk separation at the end. ###
                #
                applst(
                    contexts_tensors, permissions_tensors,
                    chunks[id_chunk].chunk_global_context_data, chunks[id_chunk].permission_mask_global_context_data
                )
                #
                applst(
                    contexts_tensors, permissions_tensors,
                    self.toks["chunk_sep"], self.permissions_vectors["separation"]
                )

        #
        context: Tensor = torch.cat( tensors=contexts_tensors, dim=-1 )
        permissions_mask: Tensor = torch.cat( tensors=permissions_tensors, dim=-2 )

        #
        if chunks_modified_hidden_states is not None and current_chunk in chunks_modified_hidden_states:

            #
            for tok_id, modified_hidden_state in chunks_modified_hidden_states[current_chunk].items():

                #
                if context.ndim == 2:
                    #
                    context[tok_id, :] = modified_hidden_state
                #
                elif context.ndim == 3:
                    #
                    context[0, tok_id, :] = modified_hidden_state

        #
        attention_causal_mask: Tensor = self.prepare_attention_causal_mask_from_permissions_mask( permissions_mask=permissions_mask )

        #
        return context, permissions_mask, attention_causal_mask, current_chunk_context_start_pos_idx


    #
    def next_token_prediction_chunks(
        self,
        chunks_documents: list[str],
        chunks_documents_idx: list[int],
        chunks: list[Chunk],
        chunks_lengths: list[int],
        current_chunk_idx: int,
        cursor_pos_in_current_chunk: int,
        chunks_modified_hidden_states: dict[int, dict[int, Tensor]],
        use_cache: bool = False,
    ) -> tuple[ Tensor, dict[int, dict[int, Tensor]] ]:  # (new_logits_at_cursor, chunks_modified_hidden_states)

        #
        ### Prepare the global context from the chunks ###
        #
        context: Tensor
        permissions_mask: Tensor
        attention_causal_mask: Tensor
        current_chunk_context_start_pos_idx: int
        #
        context, permissions_mask, attention_causal_mask, current_chunk_context_start_pos_idx = self.prepare_context_and_masks_for_all_chunks(
            chunks_documents=chunks_documents,
            chunks_documents_idx=chunks_documents_idx,
            chunks=chunks,
            current_chunk=current_chunk_idx,
            chunks_modified_hidden_states=chunks_modified_hidden_states
        )

        #
        ### Calculate the current token precice idx in the context. ###
        #
        cursor_pos: int = current_chunk_context_start_pos_idx + cursor_pos_in_current_chunk

        #
        ### Update the permissions mask to indicate we want to predict a certain token. ###
        #
        if permissions_mask.ndim == 2:
            #
            permissions_mask[cursor_pos, :] = self.permissions_vectors["next_token_prediction_cursor"]
        #
        elif permissions_mask.ndim == 3:
            #
            permissions_mask[0, cursor_pos, :] = self.permissions_vectors["next_token_prediction_cursor"]

        #
        ### Forward the model to get logits and hidden_state outputs. ###
        #
        logits, hidden_state = self.model.forward(
            input_ids=context,
            permissions_mask=permissions_mask,
            attention_causal_mask=attention_causal_mask,
            use_cache=use_cache
        )

        #
        ### Get the new logits to return. ###
        #
        new_logits_at_cursor: Tensor = logits[cursor_pos]

        #
        ### Update the chunks modified hidden_states. ###
        #
        if current_chunk_idx not in chunks_modified_hidden_states:
            #
            chunks_modified_hidden_states[current_chunk_idx] = {}
        #
        chunks_modified_hidden_states[current_chunk_idx][cursor_pos] = hidden_state[cursor_pos]

        #
        ### Return results and updated variables. ###
        #
        return (new_logits_at_cursor, chunks_modified_hidden_states)


    #
    def next_token_prediction_logits(
        self,
        text: str,
        documents: Optional[dict[str, str]] = None,
        max_length: int = 128,
        stop_if_eos_token: bool = True,
    ) -> Tensor:

        #
        ### Split text into chunks. ###
        #
        chunks_documents: list[str]
        chunks_documents_idx: list[int]
        chunks: list[Chunk]
        chunks_lengths: list[int]
        #
        chunks_documents, chunks_documents_idx, chunks, chunks_lengths = self.split_text_in_chunks(
            text=text,
            documents=documents
        )
        #
        chunks = self.init_all_chunks_global_context_with_chunk_encoding(chunks=chunks)
        #
        chunks_modified_hidden_states: dict[int, dict[int, Tensor]] = {}

        #
        all_new_logits: list[Tensor] = []
        #
        nb_generated_tokens: int = 0

        #
        ### Because we do next token prediction
        #
        current_chunk_idx: int = len(chunks) - 1
        cursor_pos_in_current_chunk: int = chunks_lengths[current_chunk_idx]
        #
        while nb_generated_tokens < max_length:

            #
            if cursor_pos_in_current_chunk >= self.model.config.chunk_length:

                #
                ### Update the global encoding of the previous chunk and create a new chunk. ###
                #
                chunks[current_chunk_idx].chunk_global_context_data = self.encode_one_chunk(chunks[current_chunk_idx])  # type: ignore
                #
                chunks_documents_idx.append( len(chunks_documents) - 1 )
                chunks.append( self.create_chunk_from_list_of_tokens(chunk_tok_ids = []) )  # type: ignore
                chunks_lengths.append( 0 )  # type: ignore
                #
                current_chunk_idx += 1
                cursor_pos_in_current_chunk = 0

            #
            ### Predict next logits. ###
            #
            new_logits_at_cursor: Tensor
            #
            new_logits_at_cursor, chunks_modified_hidden_states = self.next_token_prediction_chunks(
                chunks_documents = chunks_documents,
                chunks_documents_idx = chunks_documents_idx,
                chunks = chunks,
                chunks_lengths = chunks_lengths,
                current_chunk_idx = current_chunk_idx,
                chunks_modified_hidden_states=chunks_modified_hidden_states,
                cursor_pos_in_current_chunk = cursor_pos_in_current_chunk,
            )
            #
            all_new_logits.append( new_logits_at_cursor )
            #
            if stop_if_eos_token:
                #
                max_tok_id: int = int( torch.argmax(new_logits_at_cursor).item() )
                #
                if max_tok_id == self.model.config.tokenizer_eos_token:
                    #
                    break

            #
            ### Go to next token of the current chunk. ###
            #
            cursor_pos_in_current_chunk += 1

        #
        ### Concatenate all the previously predicted logits. ###
        #
        logits: Tensor = torch.cat(tensors=all_new_logits, dim=-2)

        #
        return logits


    #
    def next_token_prediction(
        self,
        text: str,
        documents: Optional[dict[str, str]] = None,
        max_length: int = 128,
        stop_if_eos_token: bool = True,
    ) -> Tensor:

        #
        logits: Tensor = self.next_token_prediction_logits(
            text = text,
            documents = documents,
            max_length = max_length,
            stop_if_eos_token = stop_if_eos_token,
        )

        #
        ### For the moment, just use argmax to get the tokens idx from logits, but there should be ways to improve that (example: check for better tokens that follows each others). ###
        #
        token_ids: Tensor = torch.argmax(input=logits, dim=-2)
        #
        return self.tokenizer.decode(token_ids)  # type: ignore


    #
    def decode_chunk_to_text(self, chunk: Chunk, decode_only_write: bool = True) -> str:

        #
        permission_item_chunk_inside_read_only: int = self.model.config.permissions_mask_indexes["chunk_inside_read_only"]
        permission_item_chunk_inside_read_and_write: int = self.model.config.permissions_mask_indexes["chunk_inside_read_and_write"]

        #
        ### Filter only the chunks where the permission is "chunk_inside_read_and_write" if decode_only_write ELSE filter "chunk_inside_read_only" AND"chunk_inside_read_and_write" ###
        #
        what_to_decode: Tensor = (chunk.permission_mask_context_data[:, permission_item_chunk_inside_read_and_write] > 0.5 )
        #
        if not decode_only_write:
            #
            what_to_decode |= (chunk.permission_mask_context_data[:, permission_item_chunk_inside_read_only] > 0.5 )

        #
        return self.tokenizer.decode(token_ids=chunk.chunk_context_data[what_to_decode])  # type: ignore


    #
    def decode_all_main_chunks_into_text(
        self,
        chunks_documents: list[str],
        chunks_documents_idx: list[int],
        chunks: list[Chunk],
        decode_only_write: bool = True
    ) -> str:

        #
        chunk_txt_inversed: list[str] = []

        #
        ### Normally, the "main" document chunks are the last document's chunks. ###
        #
        correct_document_idx: int = (len(chunks_documents) - 1)
        #
        for i in range(len(chunks))[::-1]:
            #
            if chunks_documents_idx[i] != correct_document_idx:
                #
                break
            #
            chunk_txt_inversed.append(
                self.decode_chunk_to_text(
                    chunk=chunks[i],
                    decode_only_write=decode_only_write
                )
            )

        #
        return "\n".join( chunk_txt_inversed[::-1] )

