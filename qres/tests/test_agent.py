import torch

from qres.agent import (
    DQN,
    MLP,
    Attention,
    AttentionHead,
    ContentEmbedding,
    Embedding,
    LayerNorm,
    PositionalEmbedding,
    Transformer,
    TransformerBlock,
)
from qres.config import N_AMINO_ACIDS, config


def test_content_embedding():
    content_embedding = ContentEmbedding()
    batch_size = 10
    seq_len = config.seq_len
    seqs = torch.randint(0, N_AMINO_ACIDS, (batch_size, seq_len), dtype=torch.long)
    embeddings = content_embedding(seqs)
    assert embeddings.shape == (batch_size, seq_len, config.d_model)


def test_positional_embedding():
    positional_embedding = PositionalEmbedding()
    batch_size = 10
    seq_len = config.seq_len
    seqs = torch.randint(0, N_AMINO_ACIDS, (batch_size, seq_len), dtype=torch.long)
    embeddings = positional_embedding(seqs)
    assert embeddings.shape == (batch_size, seq_len, config.d_model)


def test_embedding():
    embed = Embedding()
    batch_size = 10
    state_dim = config.state_dim
    states = torch.randint(0, N_AMINO_ACIDS, (batch_size, state_dim), dtype=torch.long)
    embeddings = embed(states)
    assert embeddings.shape == (batch_size, config.seq_len, config.d_model)


def test_layer_norm():
    layer_norm = LayerNorm()
    batch_size = 10
    seq_len = config.seq_len
    x = torch.randn(batch_size, seq_len, config.d_model)
    output = layer_norm(x)
    assert output.shape == x.shape


def test_mlp():
    mlp = MLP()
    batch_size = 10
    seq_len = config.seq_len
    x = torch.randn(batch_size, seq_len, config.d_model)
    output = mlp(x)
    assert output.shape == (batch_size, seq_len, config.d_model)


def test_attention_head():
    attn_head = AttentionHead()
    batch_size = 10
    seq_len = config.seq_len
    x = torch.randn(batch_size, seq_len, config.d_model)
    output = attn_head(x)
    assert output.shape == (batch_size, seq_len, config.d_model)


def test_attention():
    attn = Attention()
    batch_size = 10
    seq_len = config.seq_len
    x = torch.randn(batch_size, seq_len, config.d_model)
    output = attn(x)
    assert output.shape == (batch_size, seq_len, config.d_model)


def test_transformer_block():
    block = TransformerBlock()
    batch_size = 10
    seq_len = config.seq_len
    x = torch.randn(batch_size, seq_len, config.d_model)
    output = block(x)
    assert output.shape == (batch_size, seq_len, config.d_model)


def test_transformer():
    transformer = Transformer()
    batch_size = 10
    seq_len = config.seq_len
    x = torch.randn(batch_size, seq_len, config.d_model)
    output = transformer(x)
    assert output.shape == (batch_size, seq_len, config.d_model)


def test_dqn():
    dqn = DQN()
    batch_size = 10
    state_dim = config.state_dim
    states = torch.randint(0, N_AMINO_ACIDS, (batch_size, state_dim), dtype=torch.long)
    output = dqn(states)
    assert output.shape == (batch_size, config.action_dim)
