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
from torch.nn import functional as F
from torch.optim import Optimizer, AdamW
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
from lib_chunked_diffusion_model_config import ChunkedDiffusionModelConfig
from lib_chunked_diffusion_model import ChunkedDiffusionSystem


#
###
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
###
#
class Trainer:

    #
    def __init__(self, encoder_model_path: str = 'Alibaba-NLP/gte-large-en-v1.5') -> None:

        #
        ### Load dataset to train on. ###
        #
        self.train_lst: list[str] = []
        self.test_lst: list[str] = []
        #
        self.train_lst, self.test_lst = prepare_dataset()

        #
        ### Prepare encoder parent model. ###
        #
        self.encoder_tokenizer: PreTrainedTokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(encoder_model_path) )  # type: ignore
        self.encoder_model: PreTrainedModel = cast(PreTrainedModel, AutoModel.from_pretrained(encoder_model_path, trust_remote_code=True) )  # type: ignore

        #
        ### Init the chunked diffusion LLM model. ###
        #
        self.cdllm: ChunkedDiffusionSystem = ChunkedDiffusionSystem(
            model_config=ChunkedDiffusionModelConfig()
        )

        #
        ### Training hyper parameters. ###
        #
        ## Learning rate and optimizer. ##
        #
        self.learning_rate: float = 1e-4
        #
        self.optimizer: Optimizer = AdamW(params=self.cdllm.parameters(), lr=self.learning_rate)
        #
        self.test_each_iterations: int = 100


    #
    def get_truth_embedding(self, text: str) -> Optional[Tensor]:

        #
        ### Tokenize the text. ###
        #
        batch_dict: BatchEncoding = self.encoder_tokenizer(text, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        #
        outputs: BaseModelOutputWithPooling = self.encoder_model(**batch_dict)
        #
        if outputs.last_hidden_state is None:
            #
            return None

        #
        embeddings: Tensor = outputs.last_hidden_state[:, 0]
        #
        print(f"DEBUG | embeddings.shape = {embeddings.shape}")

        #
        return embeddings


    #
    def forward_cdllm_embedding(self, text: str, embedding_context_length: int = 4) -> Tensor:

        #
        ### Calculate the embedding with the CDLLM model. ###
        #
        return self.cdllm.simple_encode_text(text=text, encoding_length = embedding_context_length)


    #
    def loss_fn(
        self,
        truth_embedding: Tensor,  # Dim: (1, d_E)
        cdllm_embedding: Tensor   # Dim: (k, d_E)
    ) -> Tensor:
        """
        Calculate the loss based on the distances between a single truth embedding and multiple CDLLM embeddings.

        Args:
            truth_embedding (Tensor): The ground truth embedding tensor of shape (1, d_E).
            cdllm_embedding (Tensor): The CDLLM embeddings tensor of shape (k, d_E), where k is the number of embeddings.

        Returns:
            Tensor: The final loss tensor.
        """

        #
        ### Repeat the truth embedding k times to match the shape of cdllm_embedding. ###
        ### truth_embedding: (1, d_E) -> truth_repeated: (k, d_E) ###
        #
        truth_repeated: Tensor = torch.tile(truth_embedding, (cdllm_embedding.shape[0], 1))

        #
        ### Calculate the Mean Squared Error (MSE) loss for each row. ###
        ### The reduction is set to 'none' to keep the loss for each sample separately. ###
        ### The output 'all_distances' will have a shape of (k,). ###
        #
        all_distances: Tensor = F.mse_loss(truth_repeated, cdllm_embedding, reduction='none').mean(dim=1)

        #
        ### Calculate the minimum distance among all the rows. ###
        ### This finds the closest CDLLM embedding to the truth embedding. ###
        #
        min_distance: Tensor = torch.min(all_distances)

        #
        ### Calculate the mean distance of all the rows. ###
        ### This provides a holistic view of the average distance across all CDLLM embeddings. ###
        #
        mean_distance: Tensor = torch.mean(all_distances)

        #
        ### The final loss is a combination of the mean and minimum distances. ###
        ### This encourages both the average of all embeddings and the closest one to be accurate. ###
        #
        final_loss: Tensor = mean_distance + min_distance

        #
        return final_loss


    #
    def get_loss_on_embeddings(self, text: str) -> Optional[Tensor]:

        #
        ### Get truth embedding. ###
        #
        truth_embedding = self.get_truth_embedding(text=text)
        #
        if truth_embedding is None:
            #
            return None

        #
        ### Forward cdllm embedding. ###
        #
        cdllm_embedding: Tensor = self.forward_cdllm_embedding(text=text)

        #
        ### Calculate loss. ###
        #
        loss: Tensor = self.loss_fn(truth_embedding=truth_embedding, cdllm_embedding=cdllm_embedding)

        #
        return loss


    #
    def test(self) -> float:

        #
        ### Set model to evaluation mode. ###
        #
        self.cdllm.model.eval()

        #
        losses: list[float] = []

        #
        ### Test loop. ###
        #
        with torch.no_grad():  # Disable gradient computation for testing
            #
            for _i, text in tqdm(enumerate(self.test_lst), desc="Testing..."):

                #
                ### Calculate embeddings and get loss. ###
                #
                loss: Optional[Tensor] = self.get_loss_on_embeddings(text=text)
                #
                if not loss:
                    #
                    continue
                #
                losses.append(loss.item())

        #
        ### Set model back to train mode. ###
        #
        self.cdllm.model.train()

        #
        return sum(losses) / len(losses) if losses else float("nan")


    #
    def train(self) -> None:

        #
        self.cdllm.model.train()

        #
        test_loss: float = -1

        #
        ### Training loop. ###
        #
        pbar = tqdm(total=len(self.train_lst), desc="Training...")
        #
        for i, text in enumerate(self.train_lst):

            #
            ### Zero gradients before computing new ones. ###
            #
            self.optimizer.zero_grad()

            #
            ### Calculate embeddings and get loss. ###
            #
            loss: Optional[Tensor] = self.get_loss_on_embeddings(text=text)
            #
            if not loss:
                #
                print(f"Warning: Skipping sample {i} due to invalid embeddings.")
                continue

            #
            ### Backpropagate gradients and update model weights. ###
            #
            loss.backward()  # type: ignore
            self.optimizer.step()

            #
            ### Do the tests. ###
            #
            if i % self.test_each_iterations == 0:
                #
                test_loss = self.test()

            #
            pbar.update()
            #
            pbar.set_postfix_str(s=f"train loss = {loss.item()} | test loss = {test_loss}")


#
### Main function. ###
#
def main_step1_pretraining() -> None:

    #
    trainer = Trainer()
    #
    trainer.train()
