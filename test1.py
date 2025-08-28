#
### Import modules. ###
#
from torch.nn import functional as F
from torch import Tensor
#
from lib_chunked_diffusion_model_config import ChunkedDiffusionModelConfig
from lib_chunked_diffusion_model import ChunkedDiffusionSystem


#
def test_batched_encoding() -> None:

    #
    cdllm_sys: ChunkedDiffusionSystem = ChunkedDiffusionSystem(
        model_config=ChunkedDiffusionModelConfig(
            from_model_custom_config={
                "num_attention_heads": 4,
                "hidden_size": 16,
                "intermediate_size": 16*4,
                "num_hidden_layers": 4,
                "vocab_size": 128,
                "_attn_implementation": "sdpa",
            }
        )
    )

    #
    texts: list[str] = [
        "As with previous Valkyira Chronicles games , Valkyria Chronicles III is a tactical role @-@ playing game where players take control of a military unit and take part in missions against enemy forces . Stories a re told through comic book @-@ like panels with animated character portraits , with characters speaking partially through voiced speech bubbles and partially through unvoiced text . The player progresses through a series of linear missions , gradually unlocked as maps that can be freely scanned through and replayed as they are unlocked . The route to each story location on the map varies depending on an individual player 's approach : when one option is selected , the other is sealed off to the player . Outside missions , the player characters rest in a camp , where units can be customized and character growth occurs . Alongside the main story missions are character @-@ specific sub missions relating to different squad members . After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game . There are also love simulation elements related to the game 's two main heroines , although they take a very minor role . \n",
        "The game 's battle system , the BliTZ system , is carried over directly from Valkyira Chronicles . During missions , players select each unit using a top @-@ down perspective of the battlefield map : once a character is selected , the player moves the character around the battlefield in third @-@ person . A character can only act once per @-@ turn , but characters can be granted multiple turns at the expense of other characters ' turns . Each character has a field and distance of movement limited by their Action Gauge . Up to nine characters can be assigned to a single mission . During gameplay , characters will call out if something happens to them , such as their health points ( HP ) getting low or being knocked out by enemy attacks . Each character has specific \" Potentials \" , skills unique to each character . They are divided into \" Personal Potential \" , which are innate skills that remain unaltered unless otherwise dictated by the story and can either help or impede a character , and \" Battle Potentials \" , which are innate skills that remain unaltered unless otherwise dictated by the story and can either help or impede a character , and \" Battle Potentials \" , which are grown throughout the game and always grant boons to a character . To learn Battle Potentials , each character has a unique \" Masters Table \" , a grid @-@ based skill table that can be used to acquire and link different skills . Characters also have Special Abilities that grant them temporary boosts on the battlefield : Kurt can activate \" Direct Command \" and move around the battlefield without depleting his Action Point gauge , the character Reila can shift into her \" Valkyria Form \" and become invincible , while Imca can target multiple enemy units with her heavy weapon .\n",
        "The star was shining, as the space navet crashed into it.",
        "Is the butterfly one of the beautiful insect of the world? Its colored wings with unique patterns are not found anywhere else. Its a real peace of art that is destroyed by the planet decline and species destructions by the humanity."
    ]

    #
    each_text_embedding: list[Tensor] = []

    #
    encoding_length: int = 1

    #
    for t in texts:
        #
        e: Tensor = cdllm_sys.simple_encode_text(
            text=t,
            encoding_length=encoding_length
        )
        #
        each_text_embedding.append( e )

    #
    batched_embedding: list[Tensor] = cdllm_sys.batched_encode_text(
        texts=texts,
        encoding_length=encoding_length
    )

    #
    for e1, e2 in zip(each_text_embedding, batched_embedding):

        #
        print(f"\n\n- e1 = {e1}\n- e2 = {e2}\n -> {F.mse_loss(input=e1, target=e2)}")



#
if __name__ == "__main__":

    #
    test_batched_encoding()

