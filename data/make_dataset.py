import torch
import nltk
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, TensorDataset
nltk.download('punkt')

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def prepare_data_loader(file_path, batch_size=32, seq_length=50):
    content = read_file(file_path)
    tokens = word_tokenize(content)
    
    token_to_idx = {token: idx for idx, token in enumerate(set(tokens), 1)}
    token_to_idx['<pad>'] = 0
    
    encoded_tokens = [token_to_idx[token] for token in tokens if token in token_to_idx]

    num_samples = (len(encoded_tokens) - 1) // seq_length
    input_data = torch.zeros((num_samples, seq_length), dtype=torch.long)
    target_data = torch.zeros((num_samples, seq_length), dtype=torch.long)

    for i in range(num_samples):
        start_idx = i * seq_length
        end_idx = start_idx + seq_length
        input_data[i] = torch.tensor(encoded_tokens[start_idx:end_idx])
        target_data[i] = torch.tensor(encoded_tokens[start_idx + 1:end_idx + 1])

    dataset = TensorDataset(input_data, target_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, {'idx_to_token': {v: k for k, v in token_to_idx.items()}, 'token_to_idx': token_to_idx}

def get_data_loader(file_path, batch_size=64, seq_length=100):
    return prepare_data_loader(file_path, batch_size, seq_length)
