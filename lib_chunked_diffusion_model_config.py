#
### Import Modules. ###
#
from transformers import (
    PretrainedConfig
)
#
from lib_load_from_hugging_face import load_config


#
### Class Model Config. ###
#
class ChunkedDiffusionModelConfig:

    #
    def __init__(
        self,

        #
        ### **From model** parameters (import and load pretrained model from hugging face transformers). ###
        #
        ## Hugging face Model Id (ex: `Qwen/Qwen2.5-1.5B`). ##
        #
        from_model_name: str = "Qwen/Qwen2.5-0.5B",
        #
        ## Indicates model family (eg: Qwen2, Qwen3, Llama-3.2, ...). ##
        #
        from_model_family: str = "Qwen2",

        #
        ### Tokenizer parameters. ###
        #
        tokenizer_padding_side: str = "right",
        #
        ## Tokenizer padding. ##
        #
        tokenizer_pad_token: int = 0,
        tokenizer_chunk_separation_token: int = 1,
        tokenizer_documents_separation_token: int = 2,
        tokenizer_document_title_and_content_separation_token: int = 3,
        tokenizer_eos_token: int = 4,

        #
        ### **Chunk** parameters. ###
        #
        ## Non global chunk length, can be masked ##
        #
        chunk_length: int = 512,
        #
        ## Global shared context of a chunk, read & write if inside this chunk, else read only. ##
        #
        chunk_global_context_length: int = 8,


    ) -> None:

        #
        #### --- Parameters --- ####
        #

        #
        ### **From model** parameters (import and load pretrained model from hugging face transformers). ###
        #
        ## Hugging face Model Id (ex: `Qwen/Qwen2.5-1.5B`). ##
        #
        self.from_model_name: str = from_model_name
        #
        ## Indicates model family (eg: Qwen2, Qwen3, Llama-3.2, ...). ##
        #
        self.from_model_family: str = from_model_family
        #
        ## Pretrained config from the model. ##
        #
        self.from_model_config: PretrainedConfig = load_config(model_name=self.from_model_name)
        #
        ## From model useful parameters values. ##
        #
        self.from_model_config_hidden_size: int = self.from_model_config.hidden_size  # type: ignore
        self.from_model_config_voc_length: int = self.from_model_config.vocab_size  # type: ignore

        #
        ### Tokenizer parameters. ###
        #
        self.tokenizer_padding_side: str = tokenizer_padding_side
        #
        ### TODO: get correct token ids from model config. ###
        #
        self.tokenizer_pad_token: int = tokenizer_pad_token
        self.tokenizer_chunk_separation_token: int = tokenizer_chunk_separation_token
        self.tokenizer_documents_separation_token: int = tokenizer_documents_separation_token
        self.tokenizer_document_title_and_content_separation_token: int = tokenizer_document_title_and_content_separation_token
        self.tokenizer_eos_token: int = tokenizer_eos_token

        #
        ### **Chunk** parameters. ###
        #
        ## Non global chunk length, can be masked: `hidden`, `system_prompt_read_only`, `file_name_read_only`, `document_read_only`, `text_read_only`, `chunk_global_read_only`, `read_and_write_inside_chunk`, `chunk_global_read_and_write`, `global_space_read_and_write` ##
        #
        self.chunk_length: int = chunk_length
        #
        ## Global shared context of a chunk, read & write if inside this chunk, else read only. ##
        #
        self.chunk_global_context_length: int = chunk_global_context_length

        #
        ### **Permissions Mask** parameters. ###
        #
        self.permissions_mask_indexes: dict[str, int] = {
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
        }
        #
        ## Nb items of the permissions mask. ##
        #
        self.permissions_mask_nb_items: int = len(
            self.permissions_mask_indexes
        )

