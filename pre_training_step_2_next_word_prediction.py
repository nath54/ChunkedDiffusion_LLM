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
from torch.optim import Optimizer, Adam
#
from datasets import Dataset  # type: ignore
#
from tqdm import tqdm  # type: ignore
#
from lib_chunked_diffusion_model_config import ChunkedDiffusionModelConfig
from lib_chunked_diffusion_model import ChunkedDiffusionSystem
from lib_chunks import Chunk


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
        with open(".cache/filtered_dataset_cache.json") as f:
            #
            cache: dict[str, Any] = json.load(fp=f)
            #
            self.train_lst = cache["train"]
            self.test_lst = cache["test"]

        #
        self.train_lst = self.train_lst[:1000]
        self.test_lst = self.test_lst[:100]

        #
        ### Init the chunked diffusion LLM model. ###
        #
        self.cdllm: ChunkedDiffusionSystem = ChunkedDiffusionSystem(
            model_config=ChunkedDiffusionModelConfig(
                from_qlora_model=True
            )
        )

        #
        ### Training hyper parameters. ###
        #
        ## Learning rate and optimizer. ##
        #
        self.learning_rate: float = 1e-6
        #
        self.optimizer: Optimizer = Adam(params=self.cdllm.parameters(), lr=self.learning_rate)
        #
        self.test_each_iterations: int = 200
        #
        self.batch_size_train: int = 1
        self.batch_size_test: int = 1


    #
    def forward_cdllm_logits(
        self,
        text: str,
        documents: Optional[dict[str, str]] = None,
        max_length: int = 512,
        stop_if_eos_token: bool = True,
        generate_n_toks_per_n_toks: int = 16
    ) -> tuple[ list[Chunk], dict[int, list[Tensor]] ]:

        #
        ### Split text into chunks. ###
        #
        chunks_documents: list[str]
        chunks_documents_idx: list[int]
        chunks: list[Chunk]
        chunks_lengths: list[int]
        #
        chunks_documents, chunks_documents_idx, chunks, chunks_lengths = self.cdllm.prepare_chunks_for_next_tokens_predictions(
            text = text,
            documents = documents,
            max_length = max_length,
            stop_if_eos_token = stop_if_eos_token,
            generate_n_toks_per_n_toks = generate_n_toks_per_n_toks,
        )
        #
        chunks_modified_hidden_states: dict[int, dict[int, Tensor]] = {}

        #
        ### Because we do next token prediction
        #
        positions_to_generate_on_each_chunks: dict[int, list[int]] = {}
        #
        for chunk_idx, _chunk in enumerate(chunks):
            #
            if chunk_idx != 0:
                #
                positions_to_generate_on_each_chunks[chunk_idx] = list(range(0, chunks_lengths[chunk_idx]))
            #
            else:
                #
                positions_to_generate_on_each_chunks[chunk_idx] = list(range(1, chunks_lengths[chunk_idx]))

        #
        generated_logits: dict[int, list[Tensor]] = self.cdllm.next_token_prediction_logits_from_chunks_directly(
            chunks_documents=chunks_documents,
            chunks_documents_idx=chunks_documents_idx,
            chunks=chunks,
            chunks_lengths=chunks_lengths,
            positions_to_generate_on_each_chunks=positions_to_generate_on_each_chunks,
            stop_if_eos_token=stop_if_eos_token,
            generate_n_toks_per_n_toks=generate_n_toks_per_n_toks,
            chunks_modified_hidden_states=chunks_modified_hidden_states
        )

        #
        return chunks, generated_logits


    #
    def forward_cdllm_logits_batched(self, texts: list[str]) -> list[ tuple[ list[Chunk], dict[int, list[Tensor]] ] ]:

        #
        ### Fake batched for now. ###
        #
        batched_outputs: list[ tuple[ list[Chunk], dict[int, list[Tensor]] ] ] = []
        #
        for text in texts:
            #
            batched_outputs.append( self.forward_cdllm_logits(text=text) )

        #
        return batched_outputs


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
        ### Forward cdllm embedding. ###
        #
        chunks_out: list[Chunk]
        cdllm_logits: dict[int, list[Tensor]]
        #
        chunks_out, cdllm_logits = self.forward_cdllm_logits(text=text)

        #
        ### Calculate loss. ###
        #
        loss: Tensor = torch.tensor([0], device=self.cdllm.device, dtype=self.cdllm.dtype)

        #
        id_chunk: int
        logits: list[Tensor]
        #
        for id_chunk, logits in cdllm_logits.items():

            #
            chunk: Chunk = chunks_out[id_chunk]

            #
            idx_first_tensor: int = 0
            #
            if id_chunk == 0:

                #
                idx_first_tensor = 1

            #
            for idx_tensor, logit_tensor in enumerate( logits ):

                #
                loss_item: Tensor = self.loss_fn(
                    truth_tokens=chunk.chunk_context_data[idx_first_tensor + idx_tensor],
                    cdllm_logits=logit_tensor
                )

                #
                loss += loss_item

        #
        return loss


    #
    def get_loss_on_texts_batched(self, dataset_idxs: list[int], from_dataset: str = "train") -> Optional[Tensor]:

        #
        ### Calculate loss for each item in the batch. ###
        #
        loss: Tensor = torch.tensor([0], device=self.cdllm.device, dtype=self.cdllm.dtype)
        #
        for dataset_idx in dataset_idxs:

            #
            loss_item: Optional[Tensor] = self.get_loss_on_text(dataset_idx=dataset_idx, from_dataset=from_dataset)

            #
            if loss_item is not None:
                #
                loss += loss_item

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
            with torch.autograd.detect_anomaly():

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
                #
                torch.nn.utils.clip_grad_norm_(self.cdllm.parameters(), max_norm=1.0)
                #
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

