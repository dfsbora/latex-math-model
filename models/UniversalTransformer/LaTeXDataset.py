"""Handles data loading and preprocessing.

Loads and tokenizes LaTeX data.
Provides length and item access.
"""

import torch
from torch.utils.data import Dataset


class LaTeXDataset(Dataset):
    def __init__(self, filepaths, tokenizer, max_len):
        """
        Initialize the dataset with file paths and a tokenizer.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        data = [open(fp, 'r', encoding='utf-8').read() for fp in filepaths]  # Load the content of all files
        self.data = [self._truncate(tokens) for tokens in data]

    def _truncate(self, tokens):
        """
        Truncate tokens to the maximum length.
        """
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        return tokens

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)  # Return the total number of samples

    def __getitem__(self, idx):
        """
        Return the tokenized sample at index idx.
        """
        source = self.data[idx]  # Get the content of the file at the specified index
        tokenized = self.tokenizer.encode(source)  # Tokenize the content
        return torch.tensor(tokenized.ids)  # Return the tokenized ids as a tensor
