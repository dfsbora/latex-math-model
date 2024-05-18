"""Training and Evaluation Functions.

Trains the model and saves the state.
Evaluates the model and prints the loss.
Loads the model state from a file.
"""

import torch
import torch.optim as optim
from torch import nn
from torch.cuda.amp import autocast, GradScaler


# def train_model(model, data_loader, num_epochs, lr=1e-4, save_path="model.pth", use_mixed_precision=False):
#     """
#     Train the model and save the model state to save_path.
#
#     Args:
#         model: The model to be trained.
#         data_loader: The DataLoader providing the training data.
#         num_epochs: The number of epochs to train the model.
#         lr: Learning rate for the optimizer.
#         save_path: Path to save the trained model.
#         use_mixed_precision: Whether to use mixed precision training (default: False).
#
#     Note:
#         Mixed Precision Training is a technique that utilizes both 16-bit (half precision)
#         and 32-bit (single precision) floating-point numbers to train deep learning models.
#         This approach aims to reduce memory usage and increase computational efficiency
#         without compromising the accuracy or performance of the model.
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
#     model.to(device)  # Move model to device
#     model.train()  # Set model to training mode
#     criterion = nn.CrossEntropyLoss(ignore_index=0)  # Define loss function, ignoring padding index
#     optimizer = optim.Adam(model.parameters(), lr=lr)  # Define optimizer
#     scaler = GradScaler() if use_mixed_precision else None  # Initialize GradScaler if mixed precision is used
#
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for batch in data_loader:
#             source_batch = batch.to(device)  # Move source batch to device
#             target_batch = batch.to(device)  # Move target batch to device
#
#             optimizer.zero_grad()  # Zero the gradients
#
#             if use_mixed_precision:
#                 with autocast():
#                     output, _ = model(source_batch, target_batch[:, :-1])  # Forward pass
#                     loss = criterion(output.view(-1, output.size(-1)),
#                                      target_batch[:, 1:].reshape(-1))  # Calculate loss
#                 scaler.scale(loss).backward()  # Backward pass with scaling
#                 scaler.step(optimizer)  # Optimizer step with scaling
#                 scaler.update()  # Update the scaler
#             else:
#                 output, _ = model(source_batch, target_batch[:, :-1])  # Forward pass
#                 loss = criterion(output.view(-1, output.size(-1)),
#                                  target_batch[:, 1:].reshape(-1))  # Calculate loss
#                 loss.backward()  # Backward pass
#                 optimizer.step()  # Update model parameters
#
#             total_loss += loss.item()  # Accumulate loss
#
#         avg_loss = total_loss / len(data_loader)  # Calculate average loss
#         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')  # Print loss
#
#         # Save the model state
#         torch.save(model.state_dict(), save_path)

def train_model(model, data_loader, num_epochs, lr=1e-4, save_path="model.pth",
                use_mixed_precision=False, accumulation_steps=4):
    """
    Train the model and save the model state to save_path.

    Args:
        model: The model to be trained.
        data_loader: The DataLoader providing the training data.
        num_epochs: The number of epochs to train the model.
        lr: Learning rate for the optimizer.
        save_path: Path to save the trained model.
        use_mixed_precision: Whether to use mixed precision training (default: False).
        accumulation_steps: Number of batches to accumulate gradients before updating weights.

    Note:
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
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Define optimizer
    scaler = GradScaler() if use_mixed_precision else None  # Initialize GradScaler if mixed precision is used

    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()  # Zero the gradients at the start of each epoch
        accumulation_counter = 0  # Counter for gradient accumulation
        for batch_idx, batch in enumerate(data_loader):
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

        # Handle remaining gradients after the last batch of the epoch
        if accumulation_counter != 0:
            if use_mixed_precision:
                scaler.step(optimizer)  # Optimizer step with scaling
                scaler.update()  # Update the scaler
            else:
                optimizer.step()  # Update model parameters
            optimizer.zero_grad()  # Zero the gradients after accumulation

        avg_loss = total_loss / len(data_loader)  # Calculate average loss
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')  # Print loss

        # Save the model state
        torch.save(model.state_dict(), save_path)


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

    avg_loss = total_loss / len(data_loader)  # Calculate average loss
    print(f'Validation Loss: {avg_loss}')  # Print loss


def load_model(model, path):
    """
    Load the model state from path.
    """
    model.load_state_dict(torch.load(path))  # Load model state
