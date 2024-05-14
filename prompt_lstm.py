import torch
import pickle
import argparse
from models.lstm import LSTMModel
from torch.nn.functional import softmax

def load_model(model_path, vocab_size, embedding_dim, hidden_dim, num_layers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def load_vocab_mappings(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab_mappings = pickle.load(f)
    vocab_size = len(vocab_mappings["idx_to_token"])
    return vocab_mappings, vocab_size

def predict(model, vocab_mappings, initial_str, predict_len=100, temperature=1.0):
    device = next(model.parameters()).device
    
    idx_to_token = vocab_mappings['idx_to_token']
    token_to_idx = vocab_mappings['token_to_idx']
    
    hidden = model.init_hidden(1)

    input_idx = [token_to_idx[token] for token in initial_str if token in token_to_idx]
    input_tensor = torch.tensor(input_idx, dtype=torch.long).to(device).unsqueeze(0)
    
    predicted_text = initial_str
    
    for p in range(predict_len):
        output, hidden = model(input_tensor, hidden)
        
        output_dist = output.data.view(-1).div(temperature).exp()
        output_dist = output_dist / output_dist.sum()
        
        # Ensure no out-of-index selections
        top_i = torch.multinomial(output_dist, 1)[0]
        while top_i.item() >= len(idx_to_token):
            top_i = torch.multinomial(output_dist, 1)[0]
        
        predicted_char = idx_to_token[top_i.item()]
        predicted_text += predicted_char
        
        input_tensor = top_i.unsqueeze(0).unsqueeze(0)
    
    return predicted_text

def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained LSTM model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to the vocabulary mappings file.')
    parser.add_argument('--initial_str', type=str, required=True, help='Initial string to start text generation.')
    parser.add_argument('--predict_len', type=int, default=100, help='Length of the text to generate.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for prediction diversity.')

    args = parser.parse_args()

    vocab_mappings, vocab_size = load_vocab_mappings(args.vocab_path)
    model = load_model(args.model_path, vocab_size=vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2)

    output_text = predict(model, vocab_mappings, args.initial_str, args.predict_len, args.temperature)
    print("\n", output_text)

if __name__ == '__main__':
    main()