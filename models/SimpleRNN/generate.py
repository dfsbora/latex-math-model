import argparse
import os

import torch

from models.SimpleRNN.lstm import LaTeXDataset, LSTMModel, generate_text


def load_model(model_path, vocab_size, embedding_dim, hidden_dim, num_layers):
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    model.load_state_dict(torch.load(model_path))
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained LSTM model.")
    parser.add_argument('--model_path', type=str, default='./best_model.pth', help='Path to the trained model file.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing LaTeX data files.')
    parser.add_argument('--start_seq', type=str, default=r"\begin{theorem}",
                        help='The start sequence for text generation.')
    parser.add_argument('--length', type=int, default=100, help='The length of the generated text.')
    parser.add_argument('--temperature', type=float, default=0.5, help='The temperature for sampling.')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension of the LSTM model.')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of the LSTM model.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the LSTM model.')

    args = parser.parse_args()

    # Load dataset to get vocab size and mappings
    filepaths = [os.path.join(args.data_dir, fname) for fname in os.listdir(args.data_dir) if fname.endswith('.tex')]
    dataset = LaTeXDataset(filepaths)

    # Load the trained model
    model = load_model(args.model_path, dataset.vocab_size, args.embedding_dim, args.hidden_dim, args.num_layers)

    # Generate text
    generated_text = generate_text(model, dataset, args.start_seq, args.length, args.temperature)
    print(generated_text)


if __name__ == "__main__":
    main()
