# Chunked Diffusion LLM

A novel architecture implementing long context management with chunks, permission-based attention control, and textual diffusion in embedding space.

## Overview

The Chunked Diffusion LLM project introduces an innovative approach to language modeling that addresses several key challenges in modern NLP:

- **Long Context Management**: Efficient handling of extended contexts through intelligent chunking
- **Permission-based Attention**: Fine-grained control over information flow using custom permission masks. The model is aware of what token/embedding vector is where and what.
- **Textual Diffusion**: Implementation of diffusion processes directly in the embedding space
- **Multi-stage Training**: Comprehensive training pipeline for optimal model performance

## Architecture

### Core Components

1. **ChunkedDiffusionModel**: The base model that disassembles pretrained models and adds permission-based attention control. The diffusion system is not yet implemented but is coming soon.
2. **Chunk System**: Manages text chunks with both local and global contexts
3. **Permission Masks**: Controls what parts of the context can be accessed/modified

### Key Innovations

- **Dual Context System**: Each chunk has internal context and access to global context
- **Permission Vectors**: Custom permission masks concatenated with hidden states
- **Adaptive Chunking**: Intelligent text splitting to preserve semantic coherence
- **Multi-objective Training**: Combines multiple objectives for robust representation learning

## Installation

```bash
git clone https://github.com/yourusername/chunked-diffusion-llm.git
cd chunked-diffusion-llm
pip install -r requirements.txt
```

## Usage

### Basic Usage

#### Calculate text embedding

```python
from lib_chunked_diffusion_model import ChunkedDiffusionSystem
from lib_chunked_diffusion_model_config import ChunkedDiffusionModelConfig

# Initialize model configuration
config = ChunkedDiffusionModelConfig(
    from_model_name="Qwen/Qwen2.5-0.5B",
    chunk_length=512,
    chunk_global_context_length=8
)

# Create the model system
model = ChunkedDiffusionSystem(model_config=config)

# Encode text
text = "Your input text here"
embeddings = model.simple_encode_text(text=text, encoding_length=4)
```

#### Chat with next word preddiction

Not currently implemented, comming soon.

#### Chat with diffusion

Not currently implemented, comming soon.

### Training

```sh
python main.py
```

## Training Pipeline

The project implements a comprehensive 6-step training pipeline:

1. **Pre-training Step 1**: SFT encoding by learning representations from another encoding model
2. **Pre-training Step 2**: SFT diffusion denoising learning from embedding space (planned)
3. **Pre-training Step 3**: SFT next word prediction/blank filling from token space (planned)
4. **Training Step 4**: Precise document knowledge retrieval training (planned)
5. **Training Step 5**: Instruction fine-tuning from standard instruct datasets (planned)
6. **Training Step 6**: Reinforcement learning on formal benchmarks (planned)

## Configuration

The model can be configured through the `ChunkedDiffusionModelConfig` class:

```python
config = ChunkedDiffusionModelConfig(
    # Base model parameters
    from_model_name="Qwen/Qwen2.5-0.5B",
    from_model_family="Qwen2",

    # Tokenizer parameters
    tokenizer_padding_side="right",

    # Chunk parameters
    chunk_length=512,
    chunk_global_context_length=8,

    # Permission mask parameters
    permissions_mask_nb_items=9
)
```

## Supported Models

The architecture currently supports the following model families:
- Qwen (Qwen2, Qwen2.5)
- Llama
- Mistral
- GPT-2

Additional model families can be added by extending the `prepare_model` method in `ChunkedDiffusionModel`.

## Performance

The model is designed to efficiently handle long contexts while maintaining performance:

- **Memory Efficiency**: Chunking reduces memory requirements for long sequence
- **Scalability**: Architecture scales well with increasing context lengths

## Contributing

We welcome contributions! Do not hesitate to contact the developers.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{chunkeddiffusionllm,
  title={Chunked Diffusion LLM: A Novel Architecture for Textual Diffusion and Long Context Management},
  author={Nathan Cerisara},
  year={2025},
  howpublished={\url{https://github.com/nath54/ChunkedDiffusion_LLM}}
}
```

## Roadmap

- [ ] Complete implementation of diffusion in embedding space
- [ ] Add support for additional model families
- [ ] Implement reinforcement learning training stage
- [ ] Add comprehensive evaluation suite
- [ ] Explore multi-modality extensions
- [ ] Investigate mixture of experts integration

## Acknowledgments

We thank the contributors to the transformers library and the open-source AI community for their valuable tools and resources.