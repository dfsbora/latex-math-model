import argparse
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.translate.bleu_score import sentence_bleu
from models.gpt2.gpt2 import generate_text


def calculate_bleu(generated_text, reference_texts):
    """
    Calculate BLEU score for the generated text.

    :param generated_text: The generated text to evaluate.
    :param reference_texts: A list of reference texts to compare against.
    :return: BLEU score.
    """
    # Tokenize at the character level
    generated_tokens = list(generated_text)
    reference_tokens = [list(ref) for ref in reference_texts]

    # Compute BLEU score
    bleu_score = sentence_bleu(reference_tokens, generated_tokens)
    return bleu_score


def calculate_perplexity(model, tokenizer, device, generated_text):
    """
    Calculate perplexity for the generated text using the trained GPT-2 model.

    :param model: The trained GPT-2 model.
    :param tokenizer: Tokenizer to convert text to tokens.
    :param device: The device (CPU or GPU).
    :param generated_text: The generated text to evaluate.
    :return: Perplexity score.
    """
    model.eval()

    # Tokenize the input text and send to the appropriate device
    tokens = tokenizer.encode(generated_text, return_tensors="pt").to(device)

    with torch.no_grad():
        # Directly pass the tokenized input to the model to calculate the loss
        outputs = model(tokens, labels=tokens)
        loss = outputs.loss  # This already gives the average loss over all tokens

        # Perplexity is the exponent of the loss
        perplexity = torch.exp(loss)

    return perplexity.item()


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained GPT-2 model.")
    parser.add_argument('--model_dir', type=str, default='./results/final_model', help='Path to the trained model directory.')
    parser.add_argument('--start_seq', type=str, default=r"\begin{theorem}", help='The start sequence for text generation.')
    parser.add_argument('--length', type=int, default=100, help='The length of the generated text.')
    parser.add_argument('--temperature', type=float, default=0.5, help='The temperature for sampling.')
    parser.add_argument('--top_k', type=int, default=50, help='The top_k value for sampling.')
    parser.add_argument('--reference_text', type=str, help='Reference text for BLEU score evaluation.')

    args = parser.parse_args()

    # Load the trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generate text
    generated_text = generate_text(model, tokenizer, args.start_seq, args.length, args.temperature, args.top_k)
    print(f"\nGenerated Text:\n{generated_text}")

    # Calculate BLEU score if reference text is provided
    if args.reference_text:
        with open(args.reference_text, 'r') as ref_file:
            reference_text = ref_file.read().strip()
            print(f"\nReference Text:\n{reference_text}")
            bleu_score = calculate_bleu(generated_text, [reference_text])
            print(f"\nBLEU Score: {bleu_score:.4f}")

    # Calculate Perplexity
    perplexity = calculate_perplexity(model, tokenizer, device, generated_text)
    print(f"Perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    main()
