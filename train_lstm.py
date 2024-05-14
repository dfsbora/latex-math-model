import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import os
from datetime import datetime
import argparse
from models.lstm import LSTMModel
from data.make_dataset import get_data_loader


from models.lstm import LSTMModel
from data.make_dataset import get_data_loader

def train(file_path, batch_size, seq_length, epochs, embedding_dim, hidden_dim, num_layers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader, vocab_mappings = get_data_loader(file_path, batch_size, seq_length)
    vocab_size = len(vocab_mappings['token_to_idx'])

    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            actual_batch_size = inputs.size(0) 
            if actual_batch_size != batch_size:
                h = model.init_hidden(actual_batch_size)
            else:
                h = model.init_hidden(batch_size)

            inputs, targets = inputs.to(device), targets.to(device)
            h = tuple([e.data for e in h])

            model.zero_grad()
            output, h = model(inputs, h)

            loss = criterion(output, targets.view(-1))
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    print("Training complete.")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = f'lstm_model_{timestamp}.pth'
    vocab_path = f'lstm_vocab_mappings_{timestamp}.pkl'
    
    torch.save(model.state_dict(), model_path)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_mappings, f)
    print(f"Model and vocab mappings saved: '{model_path}', '{vocab_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a LSTM model")
    parser.add_argument('--data_path', type=str, default='data/data.tex', help='Path to the data file.')
    args = parser.parse_args()

    print("Training data: ", args.data_path)
    train(file_path=args.data_path, batch_size=64, seq_length=100, epochs=20,
          embedding_dim=256, hidden_dim=512, num_layers=2)