"""Orchestrates the tokenizer building/loading, data loader creation, model initialization, training, and evaluation."""

import torch
import wandb
from torch.utils.data import DataLoader

from models.UniversalTransformer.LaTeXDataset import LaTeXDataset
from models.UniversalTransformer.LaTeXTokenizer import LaTeXTokenizer
from models.UniversalTransformer.UniversalTransformer import UniversalTransformer
from models.UniversalTransformer.utils import generate_text

from models.UniversalTransformer.train import (
    train_model,
    load_model,
    evaluate_model
)


def main():
    data_dir = "data"  # Path to the directory containing LaTeX data
    tokenizer_path = "tokenizer.json"  # Path to save/load the tokenizer
    # model_save_path = "universal_transformer.pth"  # Path to save the model
    model_save_path = "checkpoint.pth.tar"  # Path to save checkpoints
    # Transformer models have a memory usage that scales quadratically with the sequence length,
    # which can (and will!) lead to out-of-memory errors for very long sequences.
    max_len = 4096  # 256; Maximum sequence length

    # Initialize the tokenizer
    tokenizer = LaTeXTokenizer(data_dir, tokenizer_path)

    # Load data and create data loader
    # batch_size = 32
    batch_size = 4  # Limited by available GPU memory
    filepaths = tokenizer.load_data()  # Load .tex file paths
    dataset = LaTeXDataset(filepaths, tokenizer.tokenizer, max_len)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0))

    # Model parameters
    source_vocab_size = len(tokenizer.tokenizer.get_vocab())  # Get source vocabulary size
    target_vocab_size = source_vocab_size  # Set target vocabulary size to be the same as source
    d_model = 1024  # Model dimension
    n_heads = 16  # Number of attention heads
    d_feedforward = 4096  # Feedforward layer dimension
    max_seq_len = max_len  # Maximum sequence length (for training and positional encoding)
    max_time_step = 10  # Maximum time steps for ACT
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

    num_epochs = 1000  # Number of epochs to train (should be divisible by save_freq for checkpoints to work)
    learning_rate = 1e-4  # Initial learning rate
    accumulation_steps = 4  # Gradient accumulation to simulate larger batch size
    use_mixed_precision = True  # Mixed precision training to save memory and speed up

    # Initialize Weights and Biases
    wandb.init(
        project="math_latex_project",
        config={
            "architecture": "Universal Transformer",
            "dataset": "The Stacks Project",

            "d_model": d_model,
            "n_heads": n_heads,
            "d_feedforward": d_feedforward,
            "max_seq_len": max_seq_len,
            "max_time_step_act": max_time_step,
            "halting_thresh_act": halting_thresh,

            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
        }
    )

    # Train the model
    train_model(model, train_loader, val_loader, tokenizer.tokenizer, num_epochs, lr=learning_rate,
                save_path=model_save_path, use_mixed_precision=use_mixed_precision, accumulation_steps=accumulation_steps)

    wandb.finish()

    # Load the model (for evaluation or further training)
    load_model(model, model_save_path)

    # Evaluate the model
    evaluate_model(model, val_loader)

    # Example of generating text
    start_sequence = r"\begin{theorem}"
    generated_text = generate_text(model, tokenizer.tokenizer, start_sequence, max_length=100, temperature=1.0,
                                   top_k=0, top_p=0.0)
    print(generated_text)


if __name__ == "__main__":
    main()
