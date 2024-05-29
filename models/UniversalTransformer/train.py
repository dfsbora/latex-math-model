"""Training and Evaluation Functions.

Trains the model and saves the state.
Evaluates the model and prints the loss.
Loads the model state from a file.
"""
import os

import torch
import torch.optim as optim
import wandb
from torch import nn
from torch.cuda.amp import autocast, GradScaler

from models.UniversalTransformer.utils import generate_text


def train_model(model, train_loader, val_loader, tokenizer, num_epochs, lr=1e-4, save_path="model.pth.tar",
                best_model_save_path="bestUniversalTransformer.pth", save_freq=5, use_mixed_precision=False, accumulation_steps=4):
    """
    Train the model and save the model state to save_path.

    Args:
        model: The model to be trained.
        train_loader: The DataLoader providing the training data.
        val_loader: The DataLoader providing the validation data.
        tokenizer: Tokenizer used for encoding/decoding.
        num_epochs: The number of epochs to train the model.
        lr: Learning rate for the optimizer.
        save_path: Path to save the trained model.
        best_model_save_path: Path to save the best model.
        save_freq: Frequency of saving checkpoints (in epochs).
        use_mixed_precision: Whether to use mixed precision training (default: False).
        accumulation_steps: Number of batches to accumulate gradients before updating weights.

    Optimization techniques applied (to be able to run the model on a gaming laptop):
        Mixed Precision Training is a technique that utilizes both 16-bit (half precision)
        and 32-bit (single precision) floating-point numbers to train deep learning models.
        This approach aims to reduce memory usage and increase computational efficiency
        without compromising the accuracy or performance of the model.

        Gradient accumulation is a technique used to simulate a larger batch size by accumulating
        gradients over several smaller batches before performing a weight update.
        This approach is particularly useful when the available GPU memory is insufficient
        to hold a large batch. By accumulating gradients over multiple smaller batches,
        one can achieve the benefits of a larger batch size without the need for excessive memory.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    model.to(device)  # Move model to device
    model.train()  # Set model to training mode
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Define loss function, ignoring padding index
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    scaler = GradScaler() if use_mixed_precision else None  # Initialize GradScaler if mixed precision is used

    best_val_loss = float('inf')
    start_epoch = 0

    # Load checkpoint if exists
    checkpoint = load_checkpoint(save_path)
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        model.train()
        optimizer.zero_grad()  # Zero the gradients at the start of each epoch
        accumulation_counter = 0  # Counter for gradient accumulation
        for batch_idx, batch in enumerate(train_loader):
            source_batch = batch.to(device)  # Move source batch to device
            target_batch = batch.to(device)  # Move target batch to device

            if use_mixed_precision:
                with autocast():
                    output, _ = model(source_batch, target_batch[:, :-1])  # Forward pass
                    loss = criterion(output.view(-1, output.size(-1)),
                                     target_batch[:, 1:].reshape(-1))  # Calculate loss
                scaler.scale(loss).backward()  # Backward pass with scaling
            else:
                output, _ = model(source_batch, target_batch[:, :-1])  # Forward pass
                loss = criterion(output.view(-1, output.size(-1)), target_batch[:, 1:].reshape(-1))  # Calculate loss
                loss.backward()  # Backward pass

            accumulation_counter += 1  # Increment the counter

            # Accumulate gradients
            if accumulation_counter % accumulation_steps == 0:
                if use_mixed_precision:
                    scaler.step(optimizer)  # Optimizer step with scaling
                    scaler.update()  # Update the scaler
                else:
                    optimizer.step()  # Update model parameters
                optimizer.zero_grad()  # Zero the gradients after accumulation
                accumulation_counter = 0  # Reset the counter

            total_loss += loss.item()  # Accumulate loss

            # Log the loss to wandb
            wandb.log({"loss_by_epoch": loss.item()})

        # Handle remaining gradients after the last batch of the epoch
        if accumulation_counter != 0:
            if use_mixed_precision:
                scaler.step(optimizer)  # Optimizer step with scaling
                scaler.update()  # Update the scaler
            else:
                optimizer.step()  # Update model parameters
            optimizer.zero_grad()  # Zero the gradients after accumulation

        # Validation step
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for source_batch in val_loader:
                source_batch = source_batch.to(device)
                target_batch = source_batch
                output, _ = model(source_batch, target_batch[:, :-1])
                loss = criterion(output.view(-1, output.size(-1)), target_batch[:, 1:].reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_loss = total_loss / len(train_loader)  # Calculate average loss
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss}, Val Loss: {avg_val_loss}')

        # Log the average loss to wandb
        wandb.log({"loss": avg_loss, "loss_val": avg_val_loss})

        # Save model checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_save_path)
            print(f"Best model saved at epoch {epoch + 1}")

        if epoch % save_freq == 0 or avg_val_loss < best_val_loss:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, save_path)

        scheduler.step(avg_val_loss)

        # Log a sample output
        if epoch % 5 == 0:  # Adjust frequency as needed
            sample_output = generate_text(model, tokenizer, r"\begin{theorem}", max_length=100)
            wandb.log({"sample_output": wandb.Html(sample_output)})


def evaluate_model(model, data_loader):
    """
    Evaluate the model on the validation set.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    model.to(device)  # Move model to device
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Define loss function, ignoring padding index

    with torch.no_grad():  # Disable gradient calculation
        for batch in data_loader:
            source_batch = batch.to(device)  # Move source batch to device
            target_batch = batch.to(device)  # Move target batch to device

            output, _ = model(source_batch, target_batch[:, :-1])  # Forward pass
            loss = criterion(output.view(-1, output.size(-1)), target_batch[:, 1:].reshape(-1))  # Calculate loss

            total_loss += loss.item()  # Accumulate loss

    avg_loss = total_loss / len(data_loader)
    print(f'Validation Loss: {avg_loss}')


# def load_model(model, path):
#     """
#     Load the model state from path.
#     """
#     model.load_state_dict(torch.load(path))  # Load model state

def load_model(model, path):
    """
    Load the model state from path.

    Args:
        model: The model to load the state into.
        path: Path to the checkpoint file.

    Returns:
        checkpoint: The loaded checkpoint dictionary.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        torch.save(state, "model_best.pth.tar")


def load_checkpoint(filename):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        return checkpoint
    else:
        print(f"No checkpoint found at '{filename}'")
        return None
