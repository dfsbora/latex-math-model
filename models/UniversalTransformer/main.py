"""Orchestrates the tokenizer building/loading, data loader creation, model initialization, training, and evaluation."""

import torch
from torch.utils.data import DataLoader

from models.UniversalTransformer.LaTeXDataset import LaTeXDataset
from models.UniversalTransformer.LaTeXTokenizer import LaTeXTokenizer
from models.UniversalTransformer.UniversalTransformer import UniversalTransformer
from models.UniversalTransformer.ut_sample import generate_text

from models.UniversalTransformer.ut_train import (
    train_model,
    load_model,
    evaluate_model
)


def main():
    data_dir = "data"  # Path to the directory containing LaTeX data
    tokenizer_path = "tokenizer.json"  # Path to save/load the tokenizer
    model_save_path = "universal_transformer.pth"  # Path to save the model
    max_len = 256  # Maximum sequence length

    # Initialize the tokenizer
    tokenizer = LaTeXTokenizer(data_dir, tokenizer_path)

    # Load data and create data loader
    # batch_size = 32
    batch_size = 2
    filepaths = tokenizer.load_data()  # Load .tex file paths
    dataset = LaTeXDataset(filepaths, tokenizer.tokenizer, max_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0))

    # Model parameters
    source_vocab_size = len(tokenizer.tokenizer.get_vocab())  # Get source vocabulary size
    target_vocab_size = source_vocab_size  # Set target vocabulary size to be the same as source
    d_model = 256  # 512; Model dimension
    n_heads = 4  # 8; Number of attention heads
    d_feedforward = 1024  # 2048; Feedforward layer dimension
    max_seq_len = max_len  # 512; Maximum sequence length
    max_time_step = 4  # 8; Maximum time steps for ACT
    halting_thresh = 0.9  # Halting threshold for ACT

    # Initialize the Universal Transformer model
    model = UniversalTransformer(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_feedforward=d_feedforward,
        max_seq_len=max_seq_len,
        max_time_step=max_time_step,
        halting_thresh=halting_thresh
    )

    # TODO:
    num_epochs = 10  # Number of epochs to train

    # Train the model
    train_model(model, data_loader, num_epochs, save_path=model_save_path, use_mixed_precision=True)

    # Load the model (for evaluation or further training)
    load_model(model, model_save_path)

    # Evaluate the model
    evaluate_model(model, data_loader)

    # Example of generating text
    start_sequence = r"\begin{theorem}"
    generated_text = generate_text(model, tokenizer.tokenizer, start_sequence, max_length=100, temperature=1.0,
                                   top_k=0, top_p=0.0)
    print(generated_text)


if __name__ == "__main__":
    main()
