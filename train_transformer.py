import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import os
from datetime import datetime
import argparse
from models.transformer import TransformerModel
from data.make_dataset import get_data_loader

def train(file_path, batch_size, seq_length, epochs, embedding_dim, num_heads, num_layers, dim_feedforward):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader, vocab_mappings = get_data_loader(file_path, batch_size, seq_length)
    vocab_size = len(vocab_mappings['token_to_idx'])

    model = TransformerModel(vocab_size, embedding_dim, num_heads, num_layers, dim_feedforward, seq_length).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            model.zero_grad()
            output = model(inputs)
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            targets = targets.view(-1)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    print("Training complete.")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = f'transformer_model_{timestamp}.pth'
    vocab_path = f'transformer_vocab_mappings_{timestamp}.pkl'
    
    torch.save(model.state_dict(), model_path)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_mappings, f)
    print(f"Model and vocab mappings saved: '{model_path}', '{vocab_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Transformer model")
    parser.add_argument('--data_path', type=str, default='data/data.tex', help='Path to the data file.')
    args = parser.parse_args()

    print("Training data: ", args.data_path)
    train(file_path=args.data_path, batch_size=64, seq_length=100, epochs=12,
          embedding_dim=256, num_heads=8, num_layers=2, dim_feedforward=512)
