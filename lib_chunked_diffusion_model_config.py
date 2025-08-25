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
        from_model_name: str = "Qwen/Qwen2.5-1.5B",
        #
        ## Indicates model family (eg: Qwen2, Qwen3, Llama-3.2, ...). ##
        #
        from_model_family: str = "Qwen2",

        #
        ### Tokenizer parameters. ###
        #
        tokenizer_padding_side: str = "left",

        #
        ### **Chunk** parameters. ###
        #
        ## Non global chunk length, can be masked: `hidden`, `system prompt`, `file_name`, `read_only`, `read_and_write` ##
        #
        chunk_lenght: int = 512,
        #
        ## Global shared context of a chunk, read & write if inside this chunk, else read only. ##
        #
        chunk_global_context_length: int = 8,

        #
        ### **Mask** parameters. ###
        #
        ## Nb items of the mask. ##
        ## `hidden`, `system prompt`, `file_name`, `read_only`, `read_and_write` ##
        #
        mask_nb_items: int = 5

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
        self.from_model_config_voc_length: int = self.from_model_config.vocab_length  # type: ignore

        #
        ### Tokenizer parameters. ###
        #
        self.tokenizer_padding_side: str = tokenizer_padding_side

        #
        ### **Chunk** parameters. ###
        #
        ## Non global chunk length, can be masked: `hidden`, `system prompt`, `file_name`, `read_only`, `read_and_write` ##
        #
        self.chunk_lenght: int = chunk_lenght
        #
        ## Global shared context of a chunk, read & write if inside this chunk, else read only. ##
        #
        self.chunk_global_context_length: int = chunk_global_context_length

        #
        ### **Mask** parameters. ###
        #
        ## Nb items of the mask. ##
        ## `hidden`, `system prompt`, `file_name`, `read_only`, `read_and_write` ##
        #
        self.mask_nb_items: int = mask_nb_items

