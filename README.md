# GPT Language Model Implementation

This is a PyTorch implementation of a GPT (Generative Pre-trained Transformer) language model, based on the architecture described in the paper "Attention Is All You Need" by Vaswani et al.

## Features

- Character-level language modeling
- Multi-head self-attention mechanism
- Position embeddings
- Transformer blocks with feed-forward networks
- Layer normalization
- Dropout for regularization

## Requirements

- Python 3.7+
- PyTorch
- NumPy

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the Shakespeare dataset:
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Usage

To train and generate text:

```bash
python gpt_model.py
```

The model will:
1. Load and preprocess the Shakespeare dataset
2. Train the model for 5000 iterations
3. Generate sample text using the trained model

## Model Architecture

The model consists of:
- Token and position embeddings
- Multiple transformer blocks, each containing:
  - Multi-head self-attention
  - Feed-forward network
  - Layer normalization
- Final layer normalization and output projection

## Hyperparameters

- Batch size: 16
- Block size: 32
- Embedding dimension: 64
- Number of attention heads: 4
- Number of transformer layers: 4
- Learning rate: 1e-3
- Dropout: 0.0

