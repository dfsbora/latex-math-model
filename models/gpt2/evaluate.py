import argparse
import torch
import torch.nn.functional as F
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.translate.bleu_score import sentence_bleu
from models.gpt2.gpt2 import generate_text


def calculate_bleu(generated_text, reference_texts):
    """
    Calculate BLEU score for the generated text.
    """
    generated_tokens = list(generated_text)
    reference_tokens = [list(ref) for ref in reference_texts]
    bleu_score = sentence_bleu(reference_tokens, generated_tokens)
    return bleu_score


def calculate_perplexity(model, tokenizer, device, generated_text):
    """
    Calculate perplexity for the generated text using the trained GPT-2 model.
    """
    model.eval()

    tokens = tokenizer.encode(generated_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(tokens, labels=tokens)
        loss = outputs.loss
        perplexity = torch.exp(loss)

    return perplexity.item()


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained GPT-2 model.")
    parser.add_argument('--model_dir', type=str, default='./results/final_model', help='Path to the trained model directory.')
    parser.add_argument('--reference_text', type=str, help='Reference text for BLEU score evaluation.')

    args = parser.parse_args()

    temperatures = [0.5, 0.7, 1.0]
    # Since the model is heavier on the memory only up to 800
    lengths = [100, 200, 400, 600, 800]

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
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

                start_time = time.time()
                generated_text = generate_text(model, tokenizer, prompt, length, temperature)

                end_time = time.time()
                inference_time = end_time - start_time
                print(f"\nGenerated Text:\n{generated_text}")


                if reference_text:
                    bleu_score = calculate_bleu(generated_text, [reference_text])
                    print(f"BLEU Score: {bleu_score:.4f}")

                perplexity = calculate_perplexity(model, tokenizer, device, generated_text)
                print(f"Perplexity: {perplexity:.4f}")

                print(f"Inference Time: {inference_time:.4f} seconds")

                print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
