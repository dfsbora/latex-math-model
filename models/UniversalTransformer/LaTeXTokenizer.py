"""Handles tokenization-related tasks.

Builds and loads a BPE tokenizer.
Handles data loading for .tex files.
"""

import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


class LaTeXTokenizer:
    def __init__(self, data_dir, tokenizer_path="tokenizer.json", vocab_size=32000):
        # Initialize with the directory containing LaTeX data, path to save/load tokenizer, and vocab size
        self.data_dir = data_dir
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        self.tokenizer = self.load_or_build_tokenizer()

    def load_or_build_tokenizer(self):
        """
        Load an existing tokenizer or build a new one if not found.
        """
        if os.path.exists(self.tokenizer_path):
            # Load the tokenizer from file if it exists
            tokenizer = Tokenizer.from_file(self.tokenizer_path)
        else:
            # Build a new tokenizer if the file does not exist
            tokenizer = self.build_tokenizer()
        return tokenizer

    def build_tokenizer(self):
        """
        Build a BPE tokenizer from the data in data_dir and save it to tokenizer_path.
        """
        filepaths = self.load_data()  # Load all .tex file paths
        tokenizer = Tokenizer(models.BPE())  # Initialize a BPE tokenizer model
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # Pre-tokenize using whitespace
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])  # Define a trainer with special tokens

        def batch_iterator(filepaths, batch_size=1000):
            # Function to read files in batches
            for i in range(0, len(filepaths), batch_size):
                batch = filepaths[i:i + batch_size]
                for filepath in batch:
                    with open(filepath, "r", encoding="utf-8") as f:
                        yield f.read()

        # Train the tokenizer on the files
        tokenizer.train_from_iterator(batch_iterator(filepaths), trainer=trainer)
        tokenizer.save(self.tokenizer_path)  # Save the trained tokenizer to file
        return tokenizer

    def load_data(self):
        """
        Load all .tex files from data_dir.
        """
        return [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.tex')]  # Return list of .tex file paths
