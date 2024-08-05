import argparse

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from models.gpt2.gpt2 import generate_text


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained GPT-2 model.")
    parser.add_argument('--model_dir', type=str, default='./results/final_model', help='Path to the trained model directory.')
    parser.add_argument('--start_seq', type=str, default=r"\begin{theorem}", help='The start sequence for text generation.')
    parser.add_argument('--length', type=int, default=100, help='The length of the generated text.')
    parser.add_argument('--temperature', type=float, default=0.5, help='The temperature for sampling.')
    parser.add_argument('--top_k', type=int, default=50, help='The top_k value for sampling.')

    args = parser.parse_args()

    # Load the trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)

    # Generate text
    generated_text = generate_text(model, tokenizer, args.start_seq, args.length, args.temperature, args.top_k)
    print(generated_text)


if __name__ == "__main__":
    main()
