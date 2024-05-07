from nltk.tokenize import word_tokenize
import numpy as np
import os
import matplotlib.pyplot as plt

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def count_tokens_between_eos(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Find indices of "eos" occurrences
    eos_indices = [i for i, token in enumerate(tokens) if token == "eos"]

    # Calculate token counts between subsequent "eos" occurrences
    token_counts = []
    for i in range(len(eos_indices) - 1):
        start_index = eos_indices[i] + 1
        end_index = eos_indices[i + 1]
        segment_tokens = tokens[start_index:end_index]
        token_count = len(segment_tokens)
        if isinstance(token_count, (int, float, complex)):
            token_counts.append(token_count)
        else:
            pass

    return token_counts


def stats(vector):
    vector = np.array(vector)
    mean = np.mean(vector)
    median = np.median(vector)
    print(f"Mean: {mean}\nMedian: {median}")


file = 'data.tex'
current_dir = os.path.dirname(os.path.abspath(__file__))
text_abs = os.path.join(current_dir, file)
text = read_file(text_abs)

token_counts = count_tokens_between_eos(text)
stats(token_counts)
