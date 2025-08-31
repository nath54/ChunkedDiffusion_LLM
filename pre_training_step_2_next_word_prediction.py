#
### Import modules. ###
#
from typing import Optional, Any, cast
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
from lib_dtypes import DTYPE_FLOAT


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
                from_qlora_model=False
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
    def train_or_get_loss_on_one_text(
        self,
        text: str,
        documents: Optional[dict[str, str]] = None,
        generate_n_toks_per_n_toks: int = 16,
        inter_direct_loss: bool = True,
        return_batched_loss: bool = False,
        is_training: bool = True,
    ) -> Tensor | list[float]:

        #
        ###
        #
        if is_training and (not return_batched_loss) and (not inter_direct_loss):

            #
            ### Zero gradients before computing new ones. ###
            #
            self.optimizer.zero_grad()

        #
        ###
        #
        global_loss: Tensor = torch.zeros((1,), dtype=DTYPE_FLOAT, device=self.cdllm.device)

        #
        losses: list[float] = []

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
            documents = documents
        )

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
        all_new_logits_per_chunks: dict[int, list[Tensor]] = {}

        #
        current_chunk_idx: int
        where_to_generate_inside_chunk: list[int]
        #
        for current_chunk_idx, where_to_generate_inside_chunk in positions_to_generate_on_each_chunks.items():

            #
            all_new_logits_per_chunks[current_chunk_idx] = []

            #
            for i in range(0, len(where_to_generate_inside_chunk), generate_n_toks_per_n_toks):

                #
                cursor_pos_in_current_chunk: list[int] = where_to_generate_inside_chunk[ i : (i+generate_n_toks_per_n_toks) ]

                #
                ### Predict next logits. ###
                #

                #
                ### Prepare the global context from the chunks ###
                #
                context: Tensor
                permissions_mask: Tensor
                attention_causal_mask: Tensor
                current_chunk_context_start_pos_idx: int
                embeddings_override: list[tuple[tuple[int, ...], tuple[int, ...], Tensor]]
                #
                context, permissions_mask, attention_causal_mask, current_chunk_context_start_pos_idx, embeddings_override = self.cdllm.prepare_context_and_masks_for_all_chunks(
                    chunks_documents=chunks_documents,
                    chunks_documents_idx=chunks_documents_idx,
                    chunks=chunks,
                    current_chunk=current_chunk_idx,
                    chunks_modified_hidden_states={}
                )

                #
                ### Calculate the current token precice idx in the context. ###
                #
                cursor_pos: list[int]
                #
                cursor_pos = [
                    current_chunk_context_start_pos_idx + c
                    for c in cursor_pos_in_current_chunk
                ]

                #
                ### Update the permissions mask to indicate we want to predict a certain token. ###
                #
                if permissions_mask.ndim == 2:
                    #
                    for c in cursor_pos:
                        #
                        permissions_mask[c, :] = self.cdllm.permissions_vectors["next_token_prediction_cursor"]
                #
                elif permissions_mask.ndim == 3:
                    #
                    for c in cursor_pos:
                        #
                        permissions_mask[0, c, :] = self.cdllm.permissions_vectors["next_token_prediction_cursor"]

                #                                                                       #
                #########################################################################
                #                                                                       #
                #########################################################################
                #                                                                       #

                #
                if is_training and (not return_batched_loss) and inter_direct_loss:

                    #
                    ### Zero gradients before computing new ones. ###
                    #
                    self.optimizer.zero_grad()

                #
                ### Forward the model to get logits and hidden_state outputs. ###
                #
                logits, _hidden_state = self.cdllm.model.forward(
                    input_ids=context,
                    permissions_mask=permissions_mask,
                    attention_causal_mask=attention_causal_mask,
                    use_cache=False,
                    embedding_overide=embeddings_override
                )

                #
                ### Get the new logits to return. ###
                #
                new_logits_at_cursor: list[Tensor] = []
                #
                if logits.ndim == 2:
                    #
                    for c in cursor_pos:
                        #
                        new_logits_at_cursor.append( logits[c].unsqueeze(0) )
                #
                else:
                    #
                    for c in cursor_pos:
                        #
                        new_logits_at_cursor.append( logits[0, c].unsqueeze(0) )

                #
                ### Get the truth tokens to calculate loss on it. ###
                #
                truth_tokens_lst: list[Tensor] = []
                #
                if context.ndim == 1:
                    #
                    for c in cursor_pos:
                        #
                        truth_tokens_lst.append( context[c].unsqueeze(0)  )
                #
                else:
                    #
                    for c in cursor_pos:
                        #
                        truth_tokens_lst.append( context[0, c].unsqueeze(0)  )

                #
                predicted_logits: Tensor = torch.cat(
                    tensors=new_logits_at_cursor,
                    dim=-2
                )
                #
                truth_tokens: Tensor = torch.cat(
                    tensors=truth_tokens_lst,
                    dim=-1
                )

                #
                loss_item: Tensor = self.loss_fn(
                    truth_tokens=truth_tokens,
                    cdllm_logits=predicted_logits
                )

                #
                if not return_batched_loss and inter_direct_loss:

                    #
                    if is_training:
                        #
                        ### Backpropagate gradients and update model weights. ###
                        #
                        loss_item.backward()  # type: ignore
                        #
                        torch.nn.utils.clip_grad_norm_(self.cdllm.parameters(), max_norm=1.0)
                        #
                        self.optimizer.step()
                    #
                    losses.append( loss_item.item() )

                #
                else:

                    #
                    global_loss += loss_item


                #                                                                       #
                #########################################################################
                #                                                                       #
                #########################################################################
                #

            #
            ### Update the global encoding of the previous chunk and create a new chunk. ###
            #
            chunks[current_chunk_idx].chunk_global_context_data = self.cdllm.encode_one_chunk(chunks[current_chunk_idx])  # type: ignore
            #
            chunks_documents_idx.append( len(chunks_documents) - 1 )
            chunks.append( self.cddlm.create_chunk_from_list_of_tokens(chunk_tok_ids = []) )  # type: ignore
            chunks_lengths.append( 0 )  # type: ignore


        #
        if not return_batched_loss and not inter_direct_loss:
            #
            if is_training:
                #
                ### Backpropagate gradients and update model weights. ###
                #
                global_loss.backward()  # type: ignore
                #
                torch.nn.utils.clip_grad_norm_(self.cdllm.parameters(), max_norm=1.0)
                #
                self.optimizer.step()
            #
            losses.append( global_loss.item() )

        #
        if return_batched_loss:

            #
            return global_loss

        #
        else:

            #
            return losses


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
    def train_text_from_dataset_idx(self, dataset_idx: int, from_dataset: str = "train", inter_direct_loss: bool = True) -> list[float]:

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
        return cast( list[float], self.train_or_get_loss_on_one_text(
            text=text,
            inter_direct_loss = inter_direct_loss,
            return_batched_loss = False,
            is_training = True,
        ) )


    #
    def train_text_from_dataset_idx_batched(self, dataset_idxs: list[int], from_dataset: str = "train") -> float:

        #
        ### Zero gradients before computing new ones. ###
        #
        self.optimizer.zero_grad()

        #
        ### Calculate loss for each item in the batch. ###
        #
        loss: Tensor = torch.tensor([0], device=self.cdllm.device, dtype=self.cdllm.dtype)
        #
        for dataset_idx in dataset_idxs:

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
            loss_item: Tensor | list[float] = self.train_or_get_loss_on_one_text(
                text=text,
                inter_direct_loss=False,
                return_batched_loss = True,
                is_training = True,
            )

            #
            if isinstance( loss_item, Tensor ):
                #
                loss += loss_item

        #
        ### Backpropagate gradients and update model weights. ###
        #
        loss.backward()  # type: ignore
        #
        torch.nn.utils.clip_grad_norm_(self.cdllm.parameters(), max_norm=1.0)
        #
        self.optimizer.step()

        #
        return loss.item()


    #
    def test_text_from_dataset_idx(self, dataset_idx: int, from_dataset: str = "test", inter_direct_loss: bool = True) -> list[float]:

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
        return cast( list[float], self.train_or_get_loss_on_one_text(
            text=text,
            inter_direct_loss = inter_direct_loss,
            return_batched_loss = False,
            is_training = False,
        ) )


    #
    def test_text_from_dataset_idx_batched(self, dataset_idxs: list[int], from_dataset: str = "test") -> float:

        #
        ### Calculate loss for each item in the batch. ###
        #
        loss: Tensor = torch.tensor([0], device=self.cdllm.device, dtype=self.cdllm.dtype)
        #
        for dataset_idx in dataset_idxs:

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
            loss_item: Tensor | list[float] = self.train_or_get_loss_on_one_text(
                text=text,
                inter_direct_loss=False,
                return_batched_loss = True,
                is_training = False,
            )

            #
            if isinstance( loss_item, Tensor ):
                #
                loss += loss_item

        #
        return loss.item()


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
                ### Calculate embeddings and get loss. ###
                #
                if self.batch_size_test <= 1:
                    #
                    losses += self.test_text_from_dataset_idx(dataset_idx=i, from_dataset="test")
                #
                else:
                    #
                    losses.append(
                        self.test_text_from_dataset_idx_batched(dataset_idxs=list(range(i, i+self.batch_size_test)), from_dataset="test")
                    )

                #
                pbar_test.update(n=self.batch_size_test)


        #
        ### Set model back to train mode. ###
        #
        self.cdllm.model.train()

        #
        final_loss_value: float = sum(losses) / len(losses) if losses else float(-1)

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
        train_loss_sum: float = 0
        nb_train_losses: float = 0

        #
        ### Training loop. ###
        #
        pbar_train = tqdm(total=len(self.train_lst), desc="Training...")
        #
        for i in range(0, len(self.train_lst), self.batch_size_train):

            #
            with torch.autograd.detect_anomaly():

                #
                ### Calculate embeddings and get loss. ###
                #
                if self.batch_size_train <= 1:
                    #
                    losses: list[float] = self.train_text_from_dataset_idx(dataset_idx=i, from_dataset="train")
                    #
                    train_loss_sum += sum( losses )
                    nb_train_losses += float( len(losses) )
                #
                else:
                    #
                    train_loss_sum += self.train_text_from_dataset_idx_batched(dataset_idxs=list(range(i, i+self.batch_size_train)), from_dataset="train")
                    nb_train_losses += 1

                #
                pbar_train.update(n=self.batch_size_train)


            #
            ### Do the tests. ###
            #
            if i % self.test_each_iterations == 0:
                #
                test_loss = self.test()

            #
            pbar_train.set_postfix_str(s=f"train loss = {train_loss_sum / max(1.0, nb_train_losses)} | test loss = {test_loss}")


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

