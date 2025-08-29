#
### Import Modules. ###
#
import os
#
from pre_training_step_1_encoding import main_step1_pretraining
from pre_training_step_1a_preparing_encoding_dataset import main_step1a_preparing_dataset
from pre_training_step_2_next_word_prediction import main_step2_pretraining


#
### Main function. ###
#
def main():

    """
    The objective of this project is to explore a novel architecture (Chunked Diffusion LLM), that implements multiple innovations:
        - Long context management with `Chunks`
        - Chunks have an internal context and a global context (when you are inside a chunk, you only the see global context of the other chunks)
        - A custom `permissions_mask` directly injected to the attention in the hidden state embedding space to indicate what is what, where the model can write / edit embeddings.
        - A textual diffusion approach, with the diffusion steps done within the embedding space
        - a new linear classifier that will be used to indicate when and at what strength we will diffuse the embeddings, or if we have to move to an another chunk.
        - The permissions mask will as well indicate to the model (the model is aware of the permissions associated to each token space) where it can edit / where it can move, but also used directly by the software stacked with the model to filter and mask the scores outputed by the model.

    The project is at a very early stage (started one day ago), developed by a single person, but it's growing at a fast pace, and have a clear direction to advance.
    """

    #
    ### Pre-training step 1: SFT encoding by learning representations from another encoding model. ###
    #
    #
    ## Preparing dataset... ##
    #
    if not os.path.exists(".cache/encoding_texts_dataset.json"):
        #
        main_step1a_preparing_dataset()
    #
    main_step1_pretraining()

    #
    ### Pre-training step 2: SFT next word prediction / blank filling from token space. ###
    #
    main_step2_pretraining()

    #
    ### TODO: Pre-training step 3: SFT diffusion denoising learning from embedding space. ###
    #
    pass

    #
    ### TODO: Training step 4: Precise document knowledge retrieval training. ###
    #
    pass

    #
    ### TODO: Training step 5: Instruction fine-tuning from standard instruct datasets. ###
    #
    pass

    #
    ### TODO: Training step 6: Reinforcment learning on maths / computer science / other formal models benchmarks. ###
    #
    pass

    #
    ### TODO: further ideas: Exploring multi-modality, mixture of expert, optimized token encoder / decoder, ... ###
    #
    pass


#
### Entry point of the program. ###
#
if __name__ == "__main__":

    #
    main()
