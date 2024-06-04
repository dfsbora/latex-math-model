import torch
import torch.optim as optim
import torch.nn as nn
import pickle
import os
from datetime import datetime
import argparse
from models.lstm import LSTMModel
from data.make_dataset import get_data_loader
import wandb
from utils import generate_config_dict


def train(file_path, batch_size, seq_length, epochs, embedding_dim, hidden_dim, num_layers, separate_by_eos):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader, vocab_mappings = get_data_loader(file_path, batch_size, seq_length, separate_by_eos)
    vocab_size = len(vocab_mappings['token_to_idx'])

    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        loss_epoch = 0
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

            loss_epoch += loss.item()

            if i % 100 == 0:
                print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
                wandb.log({'train/loss': loss.item()} )

        wandb.log({'train/loss_by_epoch': loss_epoch/len(dataloader)})

    print("Training complete.")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = f'lstm_model_{timestamp}.pth'
    vocab_path = f'lstm_vocab_mappings_{timestamp}.pkl'

    # Save model and vocab mapping locally
    torch.save(model.state_dict(), model_path)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_mappings, f)
    print(f"Model and vocab mappings saved: '{model_path}', '{vocab_path}'")

    # Save model and vocab_mapping to wandb
    if not args.disable_artifacts_wandb:
        print("Now saving at wandb...")
        model_artifact = wandb.Artifact('model', type='model')
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)

        vocab_artifact = wandb.Artifact('vocab_mappings', type='vocab_mappings')
        vocab_artifact.add_file(vocab_path)
        wandb.log_artifact(vocab_artifact)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a LSTM model")
    parser.add_argument('--data_path', type=str, default='data/data.tex', help='Path to the data file.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seq_length', type=int, default=100, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--embedding_dim', default=256, help='Embedding dimension')
    parser.add_argument('--hidden_dim', default=512, help='Hidden dimension')
    parser.add_argument('--num_layers', default=2, help='Number of layers')
    parser.add_argument('--separate_by_eos', action='store_true', help='Separate the sequences using eos')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--testing_wandb', action='store_true', help='Save the outputs into a different wandb project for tests')
    parser.add_argument('--disable_artifacts_wandb', action='store_true', help='Disable saving artifacts on wandb')
    args = parser.parse_args()

    # wandb initialization
    if args.disable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    if args.testing_wandb:
        project_name = "test_project"
    else:
        project_name = "math_latex_project"

    name = f'LSTM--{args.data_path[5:-4]}--seq_{args.seq_length}'

    wandb.init(
        project=project_name,
        name=name,
        config=generate_config_dict(args,"LSTM")
    )

    if not args.disable_artifacts_wandb:
        dataset_artifact = wandb.Artifact('dataset', type='dataset')
        dataset_artifact.add_file(args.data_path)
        wandb.log_artifact(dataset_artifact)

    # Training
    print("Training data: ", args.data_path)
    train(file_path=args.data_path, batch_size=args.batch_size, seq_length=args.seq_length, epochs=args.epochs,
          embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, separate_by_eos=args.separate_by_eos)

    wandb.finish()