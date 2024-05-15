import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np


class LaTeXTokenizer:
    """This tokenizer treats LaTeX elements as single tokens to retain mathematical notation integrity."""

    def __init__(self, vocab_size=8000):
        """Patterns to capture LaTeX commands, environments, special math symbols, and text."""
        self.vocab_size = vocab_size
        self.token_pattern = re.compile(
            r"""
            \\begin\{[^}]*\}.*?\\end\{[^}]*\}|  # Entire LaTeX environments as single tokens
            \\[a-zA-Z]+\*?(?:\[[^\]]+\])*(?:\{[^}]*\})*|  # Commands with optional arguments and options
            \$\$?.*?\$\$?|  # Inline and display math
            \{[^}]*\}|  # Arguments within curly braces
            \[[^\]]*\]|  # Optional arguments within brackets
            [a-zA-Z0-9]+|  # Alphanumeric tokens
            [.,;!?]+  # Punctuation tokens
            """,
            re.DOTALL | re.VERBOSE
        )
        self.token2idx = {}
        self.idx2token = {}

    def fit_on_texts(self, texts):
        tokens = [token for text in texts for token in self.tokenize(text)]
        most_common = Counter(tokens).most_common(self.vocab_size - 2)
        self.token2idx = {token: idx + 2 for idx, (token, _) in enumerate(most_common)}
        self.token2idx['<PAD>'] = 0
        self.token2idx['<UNK>'] = 1
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        print(f"Tokenizer fitted. Number of unique tokens: {len(self.token2idx)}")

    def texts_to_sequences(self, texts):
        return [[self.token2idx.get(token, 1) for token in self.tokenize(text)] for text in texts]

    def sequences_to_texts(self, sequences):
        return [' '.join([self.idx2token.get(idx, '<UNK>') for idx in sequence]) for sequence in sequences]

    def tokenize(self, text):
        # Filter out comments before tokenizing
        text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)
        return self.token_pattern.findall(text)


class LatexDataset(Dataset):
    def __init__(self, sequences, max_len):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        padded_sequence = sequence + [0] * (self.max_len - len(sequence))
        return torch.tensor(padded_sequence[:self.max_len]), torch.tensor(padded_sequence[:self.max_len])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pos_enc = self.create_positional_encoding(d_model, max_len)

    def create_positional_encoding(self, d_model, max_len):
        pos_enc = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_enc[:, 0::2] = np.sin(position * div_term)
        pos_enc[:, 1::2] = np.cos(position * div_term)
        return torch.tensor(pos_enc, dtype=torch.float32)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pos_enc.size(0):
            self.pos_enc = self.create_positional_encoding(self.d_model, seq_len)
        return x + self.pos_enc[:seq_len, :].unsqueeze(0).to(x.device)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

    def forward(self, q, k, v):
        batch_size = q.size(0)
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.depth)
        attn = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.dense(attn_output)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output = self.attention(x, x, x)
        out1 = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output))


class UniversalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ff_dim, max_position_embed, num_layers, num_steps,
                 dropout_rate=0.1):
        super(UniversalTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_position_embed)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)])
        self.num_steps = num_steps
        self.layer_norm = nn.LayerNorm(d_model)
        self.dense = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for _ in range(self.num_steps):
            for enc_layer in self.encoder_layers:
                x = enc_layer(x)
        x = self.layer_norm(x)
        return self.dense(x)


def loss_function(real, pred):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    return loss_fn(pred.transpose(1, 2), real)


def train_model(model, dataloader, optimizer, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_function(labels, outputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')


def generate_text(model, tokenizer, start_text, device, max_len=50):
    model.eval()
    tokens = tokenizer.texts_to_sequences([start_text])[0]
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    for _ in range(max_len):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            tokens.append(next_token)
            if next_token == tokenizer.token2idx['<PAD>']:  # Stop if the model predicts the padding token
                break
            input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    generated_sequence = tokenizer.sequences_to_texts([tokens])[0]
    return generated_sequence


def read_latex_files(data_dir):
    latex_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".tex"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                latex_data.append(file.read())
    return latex_data


def create_batches(sequences, max_len, batch_size):
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        if len(batch) < batch_size:
            batch.extend([0] * (batch_size - len(batch)))  # Padding the last batch
        batches.append(batch)
    return [batch[i:i + max_len] for batch in batches for i in range(0, len(batch), max_len)]


def main():
    data_dir = "data"
    latex_data = read_latex_files(data_dir)

    if not latex_data:
        print("No LaTeX files found in the data directory.")
        return

    vocab_size = 8000
    d_model = 512
    num_heads = 8
    ff_dim = 2048
    max_position_embed = 10000
    num_layers = 6
    num_steps = 3
    dropout_rate = 0.1
    learning_rate = 0.001
    batch_size = 16  # Adjust the batch size as needed
    epochs = 20
    max_len = 512  # Adjust the max length for sequences

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = LaTeXTokenizer(vocab_size)
    tokenizer.fit_on_texts(latex_data)

    # Debug: Print some tokenized examples
    print("Example tokenized sequences:")
    for i, text in enumerate(latex_data[:3]):
        print(f"Original: {text[:100]}...")
        print(f"Tokenized: {tokenizer.tokenize(text)[:10]}")

    # Flatten all texts into a single sequence
    full_text = ' '.join(latex_data)
    full_sequence = tokenizer.texts_to_sequences([full_text])[0]

    # Create batches of sequences
    batches = create_batches(full_sequence, max_len, batch_size)
    dataset = LatexDataset(batches, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UniversalTransformer(vocab_size, d_model, num_heads, ff_dim, max_position_embed, num_layers, num_steps,
                                 dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, dataloader, optimizer, device, epochs)

    start_text = "Algebraic geometry"
    generated_text = generate_text(model, tokenizer, start_text, device)
    print("Generated Text: ", generated_text)


if __name__ == "__main__":
    main()
