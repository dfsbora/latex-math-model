"""Defines the Universal Transformer model.

Implements the Universal Transformer with encoder and decoder layers.
Supports Adaptive Computation Time (ACT).
"""

import torch
import torch.nn as nn
import math
from torch import Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, embedding_method=None):
        """
        Initialize the embedding layer with either a linear or embedding lookup.
        """
        super().__init__()
        if embedding_method == "linear":
            self.embedding = nn.Linear(vocab_size, emb_size)  # Use a linear layer for embedding
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)  # Use an embedding lookup table
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        """
        Forward pass through the embedding layer.
        """
        if isinstance(self.embedding, nn.Linear):
            tokens = tokens.unsqueeze(dim=-1).float()  # Reshape and convert to float for linear layer
            return self.embedding(tokens)  # Apply linear embedding
        else:
            return self.embedding(tokens.long()) * math.sqrt(self.emb_size)  # Apply embedding lookup and scale


class PositionalTimestepEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        """
        Initialize the positional encoding and timestep encoding.

        Args:
            d_model: Dimensionality of the model.
            dropout: Dropout rate.
            max_len: Maximum length of the sequences to be processed,
                    should be large enough to handle the longest sequence in the input data.
        """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # Create a position tensor
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))  # Calculate the div_term
        pe = torch.zeros(1, max_len, d_model)  # Initialize the positional encoding matrix
        pe[0, :, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[0, :, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        self.register_buffer("pe", pe)  # Register pe as a buffer
        self.dropout = nn.Dropout(p=dropout)  # Define dropout layer

    def forward(self, x: Tensor, time_step: int) -> Tensor:
        """
        Add positional encoding and timestep encoding to the input tensor.

        Args:
            x: Tensor of shape [batch_size, seq_len, d_model].
            time_step: Current timestep.

        Returns:
            Tensor with positional and timestep encodings added.
        """
        # x = x + self.pe[:, :x.size(1)]  # Add positional encoding
        # x = x + self.pe[:, time_step]  # Add timestep encoding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]  # Add positional encoding
        x = x + self.pe[:, time_step, :].unsqueeze(1)  # Add timestep encoding
        return self.dropout(x)  # Apply dropout


class UniversalTransformer(nn.Module):
    def __init__(self, source_vocab_size: int, target_vocab_size: int, d_model: int, n_heads: int,
                 d_feedforward: int, max_seq_len: int, max_time_step: int, halting_thresh: float,
                 embedding_method: str = None, target_input_size: int = None):
        """
        Initialize the Universal Transformer with encoder, decoder, and other layers.
        """
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_feedforward, activation='gelu',
                                                        batch_first=True)  # Define encoder layer
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_feedforward, activation='gelu',
                                                        batch_first=True)  # Define decoder layer
        self.halting_layer = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())  # Define halting layer
        self.pos_encoder = PositionalTimestepEncoding(d_model, max_len=max_seq_len)  # Define positional encoding
        target_input_size = target_input_size or target_vocab_size
        self.source_tok_emb = TokenEmbedding(source_vocab_size, d_model,
                                             embedding_method)  # Define source token embedding
        self.target_tok_emb = TokenEmbedding(target_input_size, d_model,
                                             embedding_method)  # Define target token embedding
        self.generator = nn.Linear(d_model, target_vocab_size)  # Define final linear layer for generating output
        self.max_seq_len = max_seq_len
        self.max_time_step = max_time_step
        self.halting_thresh = halting_thresh

    def forward(self, source: Tensor, target: Tensor, source_padding_mask: Tensor = None,
                target_padding_mask: Tensor = None) -> Tensor:
        """
        Forward pass through the Universal Transformer.
        """
        source = self.source_tok_emb(source)  # Apply source token embedding
        target = self.target_tok_emb(target)  # Apply target token embedding
        target_mask = self.generate_subsequent_mask(target)  # Generate target mask
        memory, ponder_time = self.forward_encoder(source, source_padding_mask)  # Forward pass through encoder
        output = self.forward_decoder(memory, target, target_mask, source_padding_mask,
                                      target_padding_mask)  # Forward pass through decoder
        output = self.generator(output)  # Generate final output
        return output, ponder_time

    def forward_encoder(self, src: Tensor, src_padding_mask: Tensor = None):
        """
        Forward pass through the encoder with ACT (Adaptive Computation Time).
        """
        halting_probability = torch.zeros((*src.shape[:-1], 1), device=src.device)  # Initialize halting probability
        remainders = torch.zeros_like(halting_probability)  # Initialize remainders
        n_updates = torch.zeros_like(halting_probability)  # Initialize update counter
        ponder_time = torch.zeros_like(halting_probability)  # Initialize ponder time
        new_src = src.clone()  # Clone source tensor

        for time_step in range(self.max_time_step):
            still_running = halting_probability < self.halting_thresh  # Determine which sequences are still running
            p = self.halting_layer(new_src)  # Calculate halting probability
            new_halted = (halting_probability + p * still_running) > self.halting_thresh  # Determine which sequences will halt
            ponder_time[~new_halted] += 1  # Increment ponder time for non-halted sequences
            still_running = (halting_probability + p * still_running) <= self.halting_thresh  # Update still running sequences
            halting_probability += p * still_running  # Update halting probability
            remainders += new_halted * (1 - halting_probability)  # Update remainders for halted sequences
            halting_probability += new_halted * remainders  # Final halting probability update
            n_updates += still_running + new_halted  # Update counter
            update_weights = p * still_running + new_halted * remainders  # Calculate update weights
            new_src = self.pos_encoder(src, time_step)  # Apply positional encoding
            new_src = self.encoder_layer(new_src,
                                         src_key_padding_mask=src_padding_mask)  # Forward pass through encoder layer
            src = (new_src * update_weights) + (src * (1 - update_weights))  # Update source tensor with new values
        return src, ponder_time

    def forward_decoder(self, memory: Tensor, target: Tensor, target_mask: Tensor = None,
                        memory_padding_mask: Tensor = None, target_padding_mask: Tensor = None):
        """
        Forward pass through the decoder with ACT (Adaptive Computation Time).
        """
        halting_probability = torch.zeros((*target.shape[:-1], 1),
                                          device=target.device)  # Initialize halting probability
        remainders = torch.zeros_like(halting_probability)  # Initialize remainders
        n_updates = torch.zeros_like(halting_probability)  # Initialize update counter
        new_target = target.clone()  # Clone target tensor
        dec_ponder_time = torch.zeros_like(halting_probability)  # Initialize ponder time for decoder

        for time_step in range(self.max_time_step):
            still_running = halting_probability < self.halting_thresh  # Determine which sequences are still running
            p = self.halting_layer(new_target)  # Calculate halting probability
            new_halted = (halting_probability + p * still_running) > self.halting_thresh  # Determine which sequences will halt
            still_running = (halting_probability + p * still_running) <= self.halting_thresh  # Update still running sequences
            halting_probability += p * still_running  # Update halting probability
            remainders += new_halted * (1 - halting_probability)  # Update remainders for halted sequences
            halting_probability += new_halted * remainders  # Final halting probability update
            n_updates += still_running + new_halted  # Update counter
            update_weights = p * still_running + new_halted * remainders  # Calculate update weights
            new_target = self.pos_encoder(target, time_step)  # Apply positional encoding
            new_target = self.decoder_layer(new_target, memory, tgt_mask=target_mask,
                                            tgt_key_padding_mask=target_padding_mask,
                                            memory_key_padding_mask=memory_padding_mask)  # Forward pass through decoder layer
            target = (new_target * update_weights) + (
                        target * (1 - update_weights))  # Update target tensor with new values
            dec_ponder_time[~new_halted] += 1  # Increment ponder time for non-halted sequences
        return target

    @staticmethod
    def generate_subsequent_mask_fullsized(target):
        """
        Generate a mask to prevent attending to future positions.

        Note:
            Not memory efficient, highly likely will lead to "CUDA out of memory" error.
        """
        sz = target.size(1) if target.dim() == 3 else target.size(0)  # Get sequence length
        target_mask = (torch.triu(torch.ones((sz, sz), device=target.device)) == 1).transpose(0, 1)  # Create upper triangular matrix
        target_mask = target_mask.float().masked_fill(target_mask == 0, float("-inf")).masked_fill(target_mask == 1,
                                                                                                   float(0.0))  # Apply mask
        return target_mask

    # @staticmethod
    # def generate_subsequent_mask(target):
    #     """
    #     Generate autoregressive mask that prevents attending to future positions.
    #
    #     Creates masks in a memory-efficient way: instead of creating a full-sized mask,
    #     create it in smaller chunks if possible.
    #
    #     Args:
    #         target: Has shape [batch_size, seq_len, embedding_dim] or [seq_len, embedding_dim]
    #
    #     Returns:
    #         Tensor with shape [seq_len, seq_len]
    #     """
    #     # Determine the size of the sequence (seq_len)
    #     sz = target.size(1) if target.dim() == 3 else target.size(0)
    #
    #     # Create an upper triangular matrix filled with ones, with the diagonal set to 1 and above set to 1
    #     # This is done using torch.triu (upper triangular matrix)
    #     mask = torch.triu(torch.ones(sz, sz, device=target.device), diagonal=1).bool()
    #
    #     # Initialize a mask matrix with zeros
    #     target_mask = torch.zeros(sz, sz, device=target.device)
    #
    #     # Fill positions in the target_mask where the mask is True (upper triangular part) with negative infinity
    #     # This prevents attending to future positions in the sequence
    #     target_mask.masked_fill_(mask, float('-inf'))
    #
    #     return target_mask

    @staticmethod
    def generate_subsequent_mask(target):
        """
        Generate an upper triangular matrix for masking subsequent positions.
        This mask is used to ensure that the model does not attend to future positions in the sequence.

        Args:
            target: Tensor of shape [batch_size, seq_len] or [seq_len, embedding_dim]

        Returns:
            A mask tensor of shape [seq_len, seq_len] with float('-inf') in the upper triangular part, and 0 in the rest.
        """
        sz = target.size(1) if target.dim() == 3 else target.size(0)
        mask = torch.full((sz, sz), float('-inf'), device='cpu').tril(-1)
        return mask.to(target.device)

    @staticmethod
    def generate_subsequent_mask_algo(target):
        """
        Generate a mask to prevent attending to future positions for algorithmic tasks.
        """
        sz = target.size(1)  # Get sequence length
        target_mask = (torch.triu(torch.ones((sz, sz), device=target.device)) == 1).transpose(0, 1)  # Create upper triangular matrix
        target_mask = target_mask.float().masked_fill(target_mask == 0, float("-inf")).masked_fill(target_mask == 1,
                                                                                                   float(0.0))  # Apply mask
        target_mask = target_mask.unsqueeze(dim=1).repeat(1, target.size(0), 1)  # Repeat mask for batch size
        return target_mask
