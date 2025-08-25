#
### Import Modules. ###
#
from typing import Any, cast
#
from transformers import (
    PretrainedConfig, PreTrainedTokenizer, PreTrainedModel,
    AutoConfig, AutoTokenizer, AutoModelForCausalLM
)
#
import datasets as dts  # type: ignore


#
### Load pre-trained model config from hugging face. ###
#
def load_config(model_name: str) -> PretrainedConfig:

    #
    return cast(PretrainedConfig, AutoConfig.from_pretrained(  # type: ignore
        model_name
    ))


#
### Load pre-trained llm from hugging face. ###
#
def load_model(model_name: str) -> PreTrainedModel:

    #
    model_loading_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_name,
        "device_map": "auto",
        "use_cache": False,
    }

    #
    model: PreTrainedModel = cast(PreTrainedModel, AutoModelForCausalLM.from_pretrained(  # type: ignore
        **model_loading_kwargs
    ) )

    #
    return model


#
### Load pre-trained tokenizer from hugging face. ###
#
def load_tokenizer(model_name: str, tokenizer_padding_side: str = "right") -> PreTrainedTokenizer:

    #
    tokenizer: PreTrainedTokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(  # type: ignore
        model_name,
        padding_side = tokenizer_padding_side
    ) )

    #
    return tokenizer


#
### Load dataset
#
def load_dataset(dataset_name: str, sub_dataset_name: str = "default") -> dts.Dataset:

    #
    dataset: dts.Dataset = dts.load_dataset(dataset_name, sub_dataset_name)  # type: ignore

    #
    return dataset
