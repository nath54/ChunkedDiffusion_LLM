#
### Import modules. ###
#
from typing import cast, Optional
#
import os
import json
#
import torch
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
from lib_get_device import get_best_device


#
### Helper function to prepare the training dataset. ###
#
def prepare_dataset() -> tuple[list[str], list[str]]:

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
    return train_lst, test_lst


#
### Trainer Class for training Chunk's initial embeddings. ###
#
class DatasetPreparation:

    #
    def __init__(self, encoder_model_path: str = 'Alibaba-NLP/gte-large-en-v1.5', n_train: int = 40000, n_test: int = 1000) -> None:

        #
        self.device: str | torch.device = get_best_device()

        #
        ### Load dataset to train on. ###
        #
        self.train_lst: list[str] = []
        self.test_lst: list[str] = []
        #
        self.train_lst, self.test_lst = prepare_dataset()
        #
        self.train_lst = self.train_lst[:n_train]
        self.test_lst = self.test_lst[:n_test]

        #
        ### Prepare encoder parent model. ###
        #
        self.encoder_tokenizer: PreTrainedTokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(encoder_model_path) )  # type: ignore
        self.encoder_model: PreTrainedModel = cast(PreTrainedModel, AutoModel.from_pretrained(encoder_model_path, trust_remote_code=True).to(device=self.device) )  # type: ignore
        self.encoder_hidden_size: int = self.encoder_model.config.hidden_size


    #
    def get_truth_embedding(self, text: str) -> Optional[Tensor]:

        #
        ### Tokenize the text. ###
        #
        batch_dict: BatchEncoding = self.encoder_tokenizer(text, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(device=self.device)
        #
        outputs: BaseModelOutputWithPooling = self.encoder_model(**batch_dict)
        #
        if outputs.last_hidden_state is None:
            #
            return None

        #
        embeddings: Tensor = outputs.last_hidden_state[:, 0]

        #
        return embeddings


    #
    def prepare_dataset(self) -> None:

        #
        if not os.path.exists(".cache/"):
            #
            os.makedirs(".cache/")

        #
        if not os.path.exists(".cache/train_tensors/"):
            #
            os.makedirs(".cache/train_tensors/")

        #
        if not os.path.exists(".cache/test_tensors/"):
            #
            os.makedirs(".cache/test_tensors/")

        #
        train_texts: list[str] = []
        test_texts: list[str] = []

        #
        ### Training dataset. ###
        #
        pbar = tqdm(total=len(self.train_lst), desc="Training dataset...")
        #
        for i, text in enumerate( self.train_lst ):

            #
            t: Optional[Tensor] = self.get_truth_embedding(text=text)

            #
            pbar.update()

            #
            if t is None:
                #
                continue

            #
            train_texts.append( text )
            #
            torch.save(
                obj=t.cpu(),
                f=f".cache/train_tensors/train_{i}.pt"
            )
            #
            del t

        #
        ### Testing dataset. ###
        #
        pbar = tqdm(total=len(self.test_lst), desc="Testing dataset...")
        #
        for i, text in enumerate( self.test_lst ):

            #
            t: Optional[Tensor] = self.get_truth_embedding(text=text)

            #
            pbar.update()

            #
            if t is None:
                #
                continue

            #
            test_texts.append( text )
            #
            torch.save(
                obj=t.cpu(),
                f=f".cache/test_tensors/test_{i}.pt"
            )
            #
            del t

        #
        with open(".cache/encoding_texts_dataset.json", "w", encoding="utf-8") as f:
            #
            json.dump(
                obj={
                    "train_texts": train_texts,
                    "test_texts": test_texts,
                },
                fp = f
            )


#
### Main function. ###
#
def main_step1a_preparing_dataset() -> None:

    #
    ### Create the trainer. ###
    #
    dataset_preparator = DatasetPreparation()
    #
    ### Start the training. ###
    #
    dataset_preparator.prepare_dataset()

