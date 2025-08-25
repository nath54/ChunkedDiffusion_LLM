#
### Import modules. ###
#
from typing import cast
#
import os
import json
#
from torch import Tensor
#
from transformers import (
    AutoTokenizer, AutoModel,
    PreTrainedTokenizer, PreTrainedModel,
    BatchEncoding
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
#
from datasets import Dataset  # type: ignore
#
from tqdm import tqdm  # type: ignore
#
from lib_load_from_hugging_face import load_dataset


#
### Main function. ###
#
def main_step1_pretraining() -> None:

    #
    ### Load dataset to train on. ###
    #
    #
    train_lst: list[str] = []
    test_lst: list[str] = []
    #
    cached_data_filtered_dataset: dict[str, list[str]] = {}

    #
    cache_folder: str = ".cache/"
    #
    filtered_dataset_cache: str = f"{cache_folder}filtered_dataset_cache.json"
    #
    if os.path.exists(filtered_dataset_cache):

        #
        with open(filtered_dataset_cache, "r", encoding="utf-8") as f:
            #
            cached_data_filtered_dataset = json.load(f)
        #
        train_lst = cached_data_filtered_dataset["train"]
        test_lst = cached_data_filtered_dataset["test"]

    #
    else:

        #
        if not os.path.exists(cache_folder):
            #
            os.makedirs(cache_folder)

        #
        dataset1: Dataset = load_dataset(dataset_name = "Salesforce/wikitext", sub_dataset_name = 'wikitext-103-raw-v1')

        #
        print("Filter train dataset...")
        #
        for i in tqdm(range(len(dataset1['train']))):  # type: ignore

            #
            txt: str = cast(str, dataset1['train'][i]['text'] )

            #
            if len(txt) > 100:

                #
                train_lst.append( txt )

        #
        print("Filter test dataset...")
        #
        for i in tqdm(range(len(dataset1['test']))):  # type: ignore

            #
            txt: str = cast(str, dataset1['test'][i]['text'] )

            #
            if len(txt) > 100:

                #
                test_lst.append( txt )

        #
        cached_data_filtered_dataset = {
            "train": train_lst,
            "test": test_lst
        }
        #
        with open(filtered_dataset_cache, "w", encoding="utf-8") as f:
            #
            json.dump(obj=cached_data_filtered_dataset, fp=f)

    #
    ### Load LLM Model to train. ###
    #
    model_path = 'Alibaba-NLP/gte-large-en-v1.5'
    tokenizer: PreTrainedTokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(model_path) )  # type: ignore
    model: PreTrainedModel = cast(PreTrainedModel, AutoModel.from_pretrained(model_path, trust_remote_code=True) )  # type: ignore

    #
    ### Tokenize the input texts. ###
    #
    batch_dict: BatchEncoding = tokenizer(train_lst[0], max_length=8192, padding=True, truncation=True, return_tensors='pt')

    outputs: BaseModelOutputWithPooling = model(**batch_dict)
    #
    print(f"DEBUG | outputs = {outputs} | type(outputs) = {type(outputs)}")
    #
    if outputs.last_hidden_state is not None:
        #
        embeddings: Tensor = outputs.last_hidden_state[:, 0]
        #
        print(f"DEBUG | embeddings = {embeddings}")
