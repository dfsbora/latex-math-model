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
                wandb.log({'loss': loss.item()} )
        wandb.log({'loss_by_epoch': loss.item()})

    print("Training complete.")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = f'lstm_model_{timestamp}.pth'
    vocab_path = f'lstm_vocab_mappings_{timestamp}.pkl'

    # Save model locally
    torch.save(model.state_dict(), model_path)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_mappings, f)
    print(f"Model and vocab mappings saved: '{model_path}', '{vocab_path}'")

    # Save model to wandb
    print("Now saving at wandb...")
    model_artifact = wandb.Artifact('model', type='model')
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)

    # Save vocab_mapping to wandb
    vocab_artifact = wandb.Artifact('vocab_mappings', type='vocab_mappings')
    vocab_artifact.add_file(vocab_path)
    wandb.log_artifact(vocab_artifact)

def generate_config_dict(arguments, architecture_name='LSTM'):
    original_dict = vars(arguments)
    keys = list(original_dict.keys())
    new_keys = keys[:-1]
    new_dict = {key: original_dict[key] for key in new_keys}
    new_dict['architecture'] = architecture_name
    return new_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a LSTM model")
    parser.add_argument('--data_path', type=str, default='data/data.tex', help='Path to the data file.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seq_length', type=int, default=100, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--embedding_dim', default=256, help='Embedding dimension')
    parser.add_argument('--hidden_dim', default=512, help='Hidden dimension')
    parser.add_argument('--num_layers', default=2, help='Number of layers')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable W&B logging')
    args = parser.parse_args()

    if args.disable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    if args.data_path != "data/data.tex":
        project_name = "test_project"
    else:
        project_name = "math_latex_project"

    wandb.init(
        project=project_name,
        config=generate_config_dict(args,"LSTM")
    )

    # Save dataset to wandb
    dataset_artifact = wandb.Artifact('dataset', type='dataset')
    dataset_artifact.add_file(args.data_path)
    wandb.log_artifact(dataset_artifact)

    print("Training data: ", args.data_path)
    train(file_path=args.data_path, batch_size=args.batch_size, seq_length=args.seq_length, epochs=args.epochs,
          embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)

    wandb.finish()