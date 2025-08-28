#
### Import modules. ###
#
from typing import Optional, Any
#
import json
#
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Optimizer, AdamW
#
from datasets import Dataset  # type: ignore
#
from tqdm import tqdm  # type: ignore
#
from lib_chunked_diffusion_model_config import ChunkedDiffusionModelConfig
from lib_chunked_diffusion_model import ChunkedDiffusionSystem


#
### Trainer Class for training Chunk's initial embeddings. ###
#
class Trainer:

    #
    def __init__(self) -> None:

        #
        ### Load dataset to train on. ###
        #
        self.train_lst: list[str] = []
        self.test_lst: list[str] = []
        #
        self.train_truth_embeddings_tensors: list[Tensor] = []
        self.test_truth_embeddings_tensors: list[Tensor] = []

        #
        ### Loading test dataset. ###
        #
        with open(".cache/encoding_texts_dataset.json") as f:
            #
            cache: dict[str, Any] = json.load(fp=f)
            #
            self.train_lst = cache["train_texts"]
            self.test_lst = cache["test_texts"]
        #
        tmp: Tensor = torch.load(".cache/train_tensors/train_1.pt")
        #
        self.encoder_hidden_size: int = tmp.shape[-1]
        #
        del tmp

        #
        ### Init the chunked diffusion LLM model. ###
        #
        self.cdllm: ChunkedDiffusionSystem = ChunkedDiffusionSystem(
            model_config=ChunkedDiffusionModelConfig(
                from_model_custom_config={
                    "num_attention_heads": 4,
                    "hidden_size": 1024,
                    "intermediate_size": 4096,
                    "num_hidden_layers": 4,
                    "vocab_size": 128,
                    "_attn_implementation": "eager",
                }
            )
        )

        #
        ### Training hyper parameters. ###
        #
        ## Learning rate and optimizer. ##
        #
        self.learning_rate: float = 1e-5
        #
        self.optimizer: Optimizer = AdamW(params=self.cdllm.parameters(), lr=self.learning_rate)
        #
        self.test_each_iterations: int = 200
        #
        self.batch_size_train: int = 16
        self.batch_size_test: int = 32

        #
        ### Random projection Embedding matrixes. ###
        #
        pass

    #
    def forward_cdllm_logits(self, text: str, nb_tokens_length: int = 1) -> Tensor:

        #
        ### TODO. ###
        #
        return Tensor()


    #
    def forward_cdllm_logits_batched(self, texts: list[str], nb_tokens_length: int = 1) -> Tensor:

        #
        ### TODO ###
        #
        return Tensor()


    #
    def loss_fn(
        self,
        truth_tokens: Tensor,  # Dim: (1, seq_length)
        cdllm_logits: Tensor   # Dim: (1, seq_length, vocab_size)
    ) -> Tensor:
        """
        _description_

        Args:
            truth_tokens (Tensor): _description_
            cdllm_logits (Tensor): _description_

        Returns:
            Tensor: The final loss tensor.
        """

        #
        ### Ensure correct device. ###
        #
        truth_tokens = truth_tokens.to(device=self.cdllm.device)
        cdllm_logits = cdllm_logits.to(device=self.cdllm.device)

        #
        loss: Tensor = F.cross_entropy(input=cdllm_logits, target=truth_tokens)

        #
        return loss


    #
    def get_loss_on_text(self, dataset_idx: int, from_dataset: str = "train") -> Optional[Tensor]:

        #
        text: str
        #
        if from_dataset == "train":
            #
            text = self.train_lst[dataset_idx]
        #
        else:
            #
            text = self.test_lst[dataset_idx]

        #
        truth_tokens: Tensor = Tensor()  # TODO

        #
        ### Forward cdllm embedding. ###
        #
        cdllm_logits: Tensor = self.forward_cdllm_logits(text=text, nb_tokens_length=1)

        #
        ### Calculate loss. ###
        #
        loss: Tensor = self.loss_fn(truth_tokens=truth_tokens, cdllm_logits=cdllm_logits)

        #
        return loss


    #
    def get_loss_on_texts_batched(self, dataset_idxs: list[int], from_dataset: str = "train") -> Optional[Tensor]:

        #
        texts: list[str] = []
        #
        for dataset_idx in dataset_idxs:

            #
            if from_dataset == "train":
                #
                texts.append( self.train_lst[dataset_idx] )
            #
            else:
                #
                texts.append( self.test_lst[dataset_idx] )

        #
        truth_tokens_batchs: list[Tensor] = []

        #
        ### Forward cdllm embedding. ###
        #
        cdllm_logits_batched: Tensor = self.forward_cdllm_logits_batched(texts=texts, nb_tokens_length=1)

        #
        ### Calculate loss. ###
        #
        loss: Tensor = torch.tensor([0], device=self.cdllm.device, dtype=self.cdllm.dtype)
        #
        for truth_tokens, cdllm_logits in zip(truth_tokens_batchs, cdllm_logits_batched):
            #

            loss += self.loss_fn(truth_tokens=truth_tokens, cdllm_logits=cdllm_logits)

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
            len_test: int = len(self.test_lst)

            #
            ### Training loop. ###
            #
            pbar_test = tqdm(total=len_test, desc="Testing...")
            #
            for i in range(0, len_test, self.batch_size_test):

                #
                ### Zero gradients before computing new ones. ###
                #
                self.optimizer.zero_grad()

                #
                ### Calculate embeddings and get loss. ###
                #
                loss: Optional[Tensor] = None
                #
                if self.batch_size_test <= 1:
                    #
                    loss = self.get_loss_on_text(dataset_idx=i, from_dataset="test")
                #
                else:
                    #
                    loss = self.get_loss_on_texts_batched(dataset_idxs=list(range(i, i+self.batch_size_test)), from_dataset="test")

                #
                pbar_test.update(n=self.batch_size_test)

                #
                if loss is None:
                    #
                    print(f"Warning: Skipping sample {i} due to invalid embeddings.")
                    continue

                #
                losses.append(loss.item())

        #
        ### Set model back to train mode. ###
        #
        self.cdllm.model.train()

        #
        final_loss_value: float = sum(losses) / len(losses) if losses else float("nan")

        #
        tqdm.write(f"Test loss = {final_loss_value}")

        #
        return final_loss_value


    #
    def train(self) -> None:

        #
        self.cdllm.model.train()

        #
        test_loss: float = -1

        #
        ### Training loop. ###
        #
        pbar_train = tqdm(total=len(self.train_lst), desc="Training...")
        #
        for i in range(0, len(self.train_lst), self.batch_size_train):

            #
            ### Zero gradients before computing new ones. ###
            #
            self.optimizer.zero_grad()

            #
            ### Calculate embeddings and get loss. ###
            #
            loss: Optional[Tensor] = None
            #
            if self.batch_size_train <= 1:
                #
                loss = self.get_loss_on_text(dataset_idx=i, from_dataset="train")
            #
            else:
                #
                loss = self.get_loss_on_texts_batched(dataset_idxs=list(range(i, i+self.batch_size_train)), from_dataset="train")

            #
            pbar_train.update(n=self.batch_size_train)

            #
            if loss is None:
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
            pbar_train.set_postfix_str(s=f"train loss = {loss.item()} | test loss = {test_loss}")


#
### Main function. ###
#
def main_step2_pretraining() -> None:

    #
    ### Create the trainer. ###
    #
    trainer = Trainer()
    #
    ### Start the training. ###
    #
    trainer.train()

