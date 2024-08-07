import argparse
import os

import torch
from transformers import GPT2Tokenizer

from models.transformer.main_student import TransformerModel


def load_model_and_tokenizer(model_dir):
    # Load the tokenizer from the specified directory
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '', 'eos_token': ''})

    # Define model parameters
    vocab_size = len(tokenizer)
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    max_seq_length = 512
    pad_token_id = tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the custom transformer model
    model = TransformerModel(
        vocab_size, d_model, nhead, num_encoder_layers,
        num_decoder_layers, dim_feedforward, max_seq_length, pad_token_id
    ).to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load(os.path.join(model_dir, "final_model.pt"), map_location=device))
    model.eval()

    return model, tokenizer, device


def generate_text(model, tokenizer, device, prompt, max_length=512, repetition_penalty=1.2):
    model.eval()
    with torch.no_grad():
        tokenized_prompt = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_length, padding=True)
        sample_text = tokenized_prompt['input_ids'].squeeze(0).to(device)
        generated = sample_text.unsqueeze(0)

        for _ in range(max_length - len(sample_text)):
            tgt_mask = model.generate_square_subsequent_mask(generated.size(1)).to(device)
            src_padding_mask = (generated == model.pad_token_id).type(torch.bool).to(device)
            tgt_padding_mask = (generated == model.pad_token_id).type(torch.bool).to(device)

            outputs = model(generated, generated, tgt_mask=tgt_mask,
                            src_key_padding_mask=src_padding_mask,
                            tgt_key_padding_mask=tgt_padding_mask)
            logits = outputs[:, -1, :]
            logits = model.apply_repetition_penalty(logits, generated, repetition_penalty)
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            generated = torch.cat((generated, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        generated_text = tokenizer.decode(generated.squeeze().tolist(), skip_special_tokens=True)
        return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained Transformer model.")
    parser.add_argument('--model_dir', type=str, default='./results/final_model', help='Path to the trained model directory.')
    parser.add_argument('--output_file', type=str, default='generated_text.txt', help='File to save the generated text.')
    parser.add_argument('--start_seq', type=str, default=r"\begin{theorem}", help='The start sequence for text generation.')
    parser.add_argument('--length', type=int, default=100, help='The length of the generated text.')
    parser.add_argument('--temperature', type=float, default=0.5, help='The temperature for sampling.')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='The repetition penalty for sampling.')

    args = parser.parse_args()

    # Load the trained model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args.model_dir)

    # Generate text
    generated_text = generate_text(model, tokenizer, device, args.start_seq, args.length, args.repetition_penalty)
    print(generated_text)

    # Save the generated text to a file
    with open(args.output_file, 'w') as f:
        f.write(generated_text)

    print(f"Generated text saved to {args.output_file}")

if __name__ == "__main__":
    main()
