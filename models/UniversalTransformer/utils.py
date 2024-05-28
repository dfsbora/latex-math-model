import torch


def generate_text(model, tokenizer, start_sequence, max_length=256, temperature=1.0, top_k=0, top_p=0.0):
    """
    Generate text using the trained Universal Transformer model.

    Autoregressive generation mode. In this mode, the model generates one token at a time,
    appends it to the input sequence, and uses the updated sequence to generate the next token.
    This process is repeated until the desired length is reached or an end-of-sequence token is produced.

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

    # Set the model to evaluation mode, disabling dropout and batch normalization,
    # ensuring deterministic and consistent behavior during text generation.
    model.eval()

    # Encode the start sequence
    input_ids = tokenizer.encode(start_sequence).ids
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # Add batch dimension

    generated_ids = input_ids.copy()

    # TODO: generate text in chunks to get a longer final sequence
    # for _ in range(max_length):
    for _ in range(max_length - len(input_ids)):
        # Disable gradient calculations to save memory and computations during inference.
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
