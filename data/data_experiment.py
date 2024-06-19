from nltk.tokenize import word_tokenize
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from collections import Counter
import argparse

class LatexFileParser:
    def __init__(self, input_file):
        self.input_file = input_file
        self.current_directory = os.path.dirname(os.path.abspath(input_file))
        self.input_file_abs = os.path.join(self.current_directory, self.input_file)
        self.content = self._read_file()
        self.pattern_single = re.compile(r'(?<!\\)\$(?!\$)(.*?)(?<!\\)\$(?!\$)')
        self.pattern_double = re.compile(r'(?<!\\)\$\$(.*?)(?<!\\)\$\$', re.DOTALL)
        self.label_pattern = re.compile(r'\\label\{(.*?)\}')
        self.cite_pattern = re.compile(r'\\cite\{(.*?)\}')

    def _read_file(self):
        with open(self.input_file_abs, 'r', encoding='utf-8') as file:
            return file.read()

    def make_list_formulas(self, output_file):
        output_file_abs = os.path.join(self.current_directory, output_file)
        double_matches = self.pattern_double.findall(self.content)
        single_matches = self.pattern_single.findall(self.content)
        print(f"Found {len(single_matches) + len(double_matches)} formulas")

        double_matches = set(double_matches)
        single_matches = set(single_matches)
        with open(output_file_abs, 'w') as out_file:
            for match in double_matches:
                out_file.write(match + "\n")
            for match in single_matches:
                out_file.write(match + "\n")

        print(f"Found {len(single_matches) + len(double_matches)} unique formulas")
        return double_matches, single_matches

    def make_list_labels(self, output_file):
        output_file_abs = os.path.join(self.current_directory, output_file)
        matches = self.label_pattern.findall(self.content)
        print(f"Found {len(matches)} labels")

        matches = set(matches)
        with open(output_file_abs, 'w') as out_file:
            for match in matches:
                out_file.write(match + "\n")
        print(f"Found {len(matches)} unique labels")
        return matches

    def make_list_cites(self, output_file):
        output_file_abs = os.path.join(self.current_directory, output_file)
        matches = self.cite_pattern.findall(self.content)
        print(f"Found {len(matches)} citations")

        matches = set(matches)
        with open(output_file_abs, 'w') as out_file:
            for match in matches:
                out_file.write(match + "\n")
        print(f"Found {len(matches)} unique citations")
        return matches



class LatexFileEOSCounter:
    def __init__(self, input_file, tokenizer=word_tokenize):
        self.input_file = input_file
        self.current_directory = os.path.dirname(os.path.abspath(input_file))
        self.input_file_abs = os.path.join(self.current_directory, self.input_file)
        self.content = self._read_file()
        self.tokenizer = tokenizer
        self.token_counts = []

    def _read_file(self):
        with open(self.input_file_abs, 'r', encoding='utf-8') as file:
            return file.read()

    def count_tokens_between_eos(self):
        # Tokenize the text
        tokens = self.tokenizer(self.content)
        freq_count_tokens = Counter(tokens)
        print("\nUsing tokenizer: ", self.tokenizer)
        print("Number of tokens in the text (word_tokenize): ", len(tokens))
        print("Number of unique tokens in the text (word_tokenize): ", len(set(tokens)))
        print("Most common tokens: ", freq_count_tokens.most_common(30))

        # Find indices of "eos" occurrences
        eos_indices = [i for i, token in enumerate(tokens) if token == "eos"]

        token_counts = []

        # Calculate token counts between subsequent "eos" occurrences
        for i in range(len(eos_indices) - 1):
            start_index = eos_indices[i] + 1
            end_index = eos_indices[i + 1]
            segment_tokens = tokens[start_index:end_index]
            token_count = len(segment_tokens)
            if isinstance(token_count, (int, float, complex)):
                token_counts.append(token_count)
            else:
                pass

        self.token_counts = np.array(token_counts)
        self.stats()

    def stats(self):
        mean = np.mean(self.token_counts )
        median = np.median(self.token_counts )
        total = np.sum(self.token_counts )
        print(f"Mean: {mean}\nMedian: {median}\nTotal: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get information about the data")
    parser.add_argument('--input_file', type=bool, default='data.tex', help='Original dataset file name.')
    args = parser.parse_args()


    latex_processor = LatexFileParser('data.tex')
    double_formulas,single_formulas= latex_processor.make_list_formulas('formulas.txt')
    labels = latex_processor.make_list_labels('labels.txt')
    cites = latex_processor.make_list_cites('cites.txt')


    eos_counter = LatexFileEOSCounter('data_eos.tex', tokenizer=word_tokenize)
    eos_counter.count_tokens_between_eos()
    exit()

