import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dim_feedforward, max_seq_length):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encodings = nn.Parameter(torch.randn(max_seq_length, embedding_dim))
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encodings[:x.size(1)].clone().detach()
        x = self.transformer(x, x)
        x = self.fc_out(x)
        return x