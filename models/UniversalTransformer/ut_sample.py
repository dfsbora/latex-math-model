"""Samples from the trained Universal Transformer model.

The sampling code includes a function to generate text using the model,
leveraging techniques such as greedy decoding or beam search for generating the output sequence.
"""

import torch

from models.UniversalTransformer.LaTeXTokenizer import LaTeXTokenizer
from models.UniversalTransformer.UniversalTransformer import UniversalTransformer
from models.UniversalTransformer.ut_train import load_model


def generate_text(model, tokenizer, start_sequence, max_length=100, temperature=1.0, top_k=0, top_p=0.0):
    """
    Generate text using the Universal Transformer model.

    Args:
        model: Trained Universal Transformer model.
        tokenizer: Tokenizer used for encoding/decoding.
        start_sequence: Initial sequence to start the generation.
        max_length: Maximum length of the generated sequence.
        temperature: Sampling temperature.
        top_k: Top-K sampling.
        top_p: Top-p (nucleus) sampling.

    Returns:
        Generated text sequence.
    """
    model.eval()  # Set the model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Encode the start sequence
    input_ids = tokenizer.encode(start_sequence).ids
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # Add batch dimension

    generated_ids = input_ids.copy()

    for _ in range(max_length):
        with torch.no_grad():
            output, _ = model(input_tensor, input_tensor)

        logits = output[:, -1, :]  # Get logits of the last generated token
        logits = logits / temperature  # Apply temperature

        if top_k > 0:
            values, indices = torch.topk(logits, top_k)
            logits[logits < values[:, -1, None]] = -float('Inf')

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        generated_ids.append(next_token)

        # Update input tensor
        input_tensor = torch.tensor(generated_ids, device=device).unsqueeze(0)

        if next_token == tokenizer.token_to_id('[SEP]'):
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    data_dir = "data"
    tokenizer_path = "tokenizer.json"
    model_save_path = "universal_transformer.pth"
    max_len = 256

    tokenizer = LaTeXTokenizer(data_dir, tokenizer_path)

    # Model parameters (should match the ones used during training)
    source_vocab_size = len(tokenizer.tokenizer.get_vocab())
    target_vocab_size = source_vocab_size
    d_model = 256
    n_heads = 4
    d_feedforward = 1024
    max_seq_len = max_len
    max_time_step = 4
    halting_thresh = 0.9

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

    # Load the trained model
    load_model(model, model_save_path)

    # Example of generating text
    start_sequence = "Algebraic Geometry"
    generated_text = generate_text(
        model, tokenizer.tokenizer, start_sequence, max_length=100, temperature=1.0, top_k=0, top_p=0.0)
    print("Generated Text:\n", generated_text)


if __name__ == "__main__":
    main()
