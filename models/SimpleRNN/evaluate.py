import argparse
import os
import torch
import torch.nn.functional as F
import time
from nltk.translate.bleu_score import sentence_bleu
from models.SimpleRNN.lstm import LaTeXDataset, LSTMModel, generate_text


def load_model(model_path, vocab_size, embedding_dim, hidden_dim, num_layers):
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    model.load_state_dict(torch.load(model_path))
    return model


def calculate_bleu(generated_text, reference_texts):
    """
    Calculate BLEU score for the generated text.
    """
    generated_tokens = list(generated_text)
    reference_tokens = [list(ref) for ref in reference_texts]
    bleu_score = sentence_bleu(reference_tokens, generated_tokens)
    return bleu_score


def calculate_perplexity(model, dataset, generated_text):
    """
    Calculate perplexity for the generated text using the trained model.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokens = dataset.text_to_tensor(generated_text).unsqueeze(0).to(device)

    with torch.no_grad():
        input_seq = tokens[:, :-1]
        target_seq = tokens[:, 1:]

        batch_size = input_seq.size(0)
        hidden = model.init_hidden(batch_size)
        hidden = tuple(h.to(device) for h in hidden)

        outputs, _ = model(input_seq, hidden)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), target_seq.view(-1))

        perplexity = torch.exp(loss)

    return perplexity.item()


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained LSTM model.")
    parser.add_argument('--model_path', type=str, default='models/SimpleRNN/results/best_model.pth', help='Path to the trained model file.')
    parser.add_argument('--data_dir', type=str, default='models/SimpleRNN/data', help='Directory containing LaTeX data files.')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension of the LSTM model.')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of the LSTM model.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the LSTM model.')
    parser.add_argument('--reference_text', type=str, help='Reference text for BLEU score evaluation.')

    args = parser.parse_args()

    # Temperatures and sequence lengths to iterate over
    temperatures = [0.5, 0.7, 1.0]
    lengths = [100, 200, 400, 600, 800, 1000, 1200]

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


    filepaths = [os.path.join(args.data_dir, fname) for fname in os.listdir(args.data_dir) if fname.endswith('.tex')]
    dataset = LaTeXDataset(filepaths)
    model = load_model(args.model_path, dataset.vocab_size, args.embedding_dim, args.hidden_dim, args.num_layers)

    reference_text = None
    if args.reference_text:
        with open(args.reference_text, 'r') as ref_file:
            reference_text = ref_file.read().strip()
            print("Reference Text:\n", reference_text)
            print("\n" + "=" * 50 + "\n")

    for temperature in temperatures:
        for length in lengths:
            print(f"\n=== Temperature: {temperature}, Sequence Length: {length} ===")

            for prompt in prompts:
                print(f"\nPrompt: '{prompt}'")

                start_time = time.time()
                generated_text = generate_text(model, dataset, prompt, length, temperature)

                end_time = time.time()
                inference_time = end_time - start_time  # Total time taken to generate the text
                tokens_generated = len(generated_text.split())  # Count tokens (words) in the generated text
                inference_speed = tokens_generated / inference_time  # Tokens generated per second

                print(f"Generated Text:\n{generated_text}")


                if reference_text:
                    bleu_score = calculate_bleu(generated_text, [reference_text])
                    print(f"BLEU Score: {bleu_score:.4f}")

                perplexity = calculate_perplexity(model, dataset, generated_text)
                print(f"Perplexity: {perplexity:.4f}")

                print(f"Inference Time: {inference_time:.4f} seconds")
                print(f"Inference Speed: {inference_speed:.2f} tokens per second")

            print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
