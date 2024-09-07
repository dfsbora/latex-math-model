import argparse
import os
import torch
import torch.nn.functional as F
import time
from transformers import GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from models.transformer.main_student import TransformerModel


class LatexDataset(torch.utils.data.Dataset):
    def __init__(self, directory, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id  # Store pad_token_id for reference
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tex')]
        self.data = self.load_data()

    def preprocess_latex(self, text):
        text = re.sub(r'\\([a-zA-Z]+)', r'\\\1', text)  # Ensure backslashes are correctly tokenized
        return text

    def load_data(self):
        data = []
        for file in self.files:
            with open(file, 'r') as f:
                content = f.read()
                content = self.preprocess_latex(content)
                tokenized = self.tokenizer(content, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
                data.append(tokenized)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        input_ids = data_item['input_ids'].squeeze(0)  # Remove the batch dimension if present
        attention_mask = data_item['attention_mask'].squeeze(0)  # Ensure attention_mask is correctly shaped
        return {'input_ids': input_ids, 'attention_mask': attention_mask}


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


def calculate_bleu(generated_text, reference_texts):
    # Tokenize at the word level
    generated_tokens = list(generated_text)
    reference_tokens = [list(ref) for ref in reference_texts]
    bleu_score = sentence_bleu(reference_tokens, generated_tokens)
    return bleu_score


def calculate_perplexity(model, tokenizer, device, generated_text):
    model.eval()
    tokens = tokenizer(generated_text, return_tensors='pt', truncation=True, padding=True).input_ids.to(device)

    with torch.no_grad():
        input_seq = tokens[:, :-1]  # Input sequence (except the last token)
        target_seq = tokens[:, 1:]  # Target sequence (shifted by 1 token)

        tgt_mask = model.generate_square_subsequent_mask(input_seq.size(1)).to(device)
        src_padding_mask = (input_seq == model.pad_token_id).type(torch.bool).to(device)
        tgt_padding_mask = (input_seq == model.pad_token_id).type(torch.bool).to(device)

        outputs = model(input_seq, input_seq, tgt_mask=tgt_mask,
                        src_key_padding_mask=src_padding_mask,
                        tgt_key_padding_mask=tgt_padding_mask)
        logits = outputs.view(-1, outputs.size(-1))
        loss = F.cross_entropy(logits, target_seq.view(-1), ignore_index=tokenizer.pad_token_id)
        perplexity = torch.exp(loss)

    return perplexity.item()


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained Transformer model.")
    parser.add_argument('--model_dir', type=str, default='models/transformer/final_model', help='Path to the trained model directory.')
    parser.add_argument('--reference_text', type=str, help='Path to a reference text for BLEU score calculation.')

    args = parser.parse_args()

    # Temperatures and sequence lengths to iterate over
    temperatures = [0.5, 0.7, 1.0]
    lengths = [100, 200, 400, 600, 800, 1000, 1200]

    # Prompts to evaluate
    prompts = [
        "Assume that",
        "Let $A$",
        "Base case",
        "We need to show that",
        "Assume that for any finite free ring map $R \to S$ the ring $S$ has regular formal fibres. Let",
        "For the sake of contradiction",
        "We will prove the contrapositive",
        "Suppose there are two such elements",
        "Gauss-Bonnet Theorem for Hyperbolic Surfaces",
        "Birkhoff's Ergodic Theorem",
        "The First Law of Thermodynamics states",
        "Once upon a time, in a bright blue pond,",
        "In English, the verb in a sentence must agree with the subject in"
    ]

    # Load the trained model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args.model_dir)

    # Print the reference text once at the beginning
    reference_text = None
    if args.reference_text:
        with open(args.reference_text, 'r') as ref_file:
            reference_text = ref_file.read().strip()
            print(f"Reference Text:\n{reference_text}")
            print("\n" + "=" * 50 + "\n")

    # Loop over different temperatures, sequence lengths, and prompts
    for temperature in temperatures:
        for length in lengths:
            print(f"\n=== Temperature: {temperature}, Sequence Length: {length} ===")

            for prompt in prompts:
                print(f"\nPrompt: '{prompt}'")

                # Measure inference time for generating the text
                start_time = time.time()

                # Generate text for the current prompt, temperature, and length
                generated_text = generate_text(model, tokenizer, device, prompt, length, temperature)

                end_time = time.time()
                inference_time = end_time - start_time
                print(f"\nGenerated Text:\n{generated_text}")

                # Calculate BLEU score if reference text is provided
                if reference_text:
                    bleu_score = calculate_bleu(generated_text, [reference_text])
                    print(f"BLEU Score: {bleu_score:.4f}")

                # Calculate Perplexity
                perplexity = calculate_perplexity(model, tokenizer, device, generated_text)
                print(f"Perplexity: {perplexity:.4f}")

                # Report Inference Time
                print(f"Inference Time: {inference_time:.4f} seconds")

                print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
