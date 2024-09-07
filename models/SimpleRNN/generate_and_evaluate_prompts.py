mport argparse
import os
import torch
import torch.nn.functional as F
import nltk
from nltk.translate.bleu_score import sentence_bleu
from models.SimpleRNN.lstm import LaTeXDataset, LSTMModel, generate_text


def load_model(model_path, vocab_size, embedding_dim, hidden_dim, num_layers):
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    model.load_state_dict(torch.load(model_path))
    return model


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


def calculate_perplexity(model, dataset, generated_text):
    """
    Calculate perplexity for the generated text using the trained model.

    :param model: The trained LSTM model.
    :param dataset: The dataset object to map text to tokens.
    :param generated_text: The generated text to evaluate.
    :return: Perplexity score.
    """
    model.eval()

    # Determine the device (CPU or CUDA) and move model to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokens = dataset.text_to_tensor(generated_text).unsqueeze(0).to(device)  # Convert text to tensor and move to device

    with torch.no_grad():
        input_seq = tokens[:, :-1]  # Input sequence (except the last token)
        target_seq = tokens[:, 1:]  # Target sequence (shifted by 1 token)

        # Initialize hidden state and move it to the correct device
        batch_size = input_seq.size(0)
        hidden = model.init_hidden(batch_size)
        hidden = tuple(h.to(device) for h in hidden)  # Move hidden states to device

        # Forward pass through the model
        outputs, _ = model(input_seq, hidden)

        # Calculate loss (cross-entropy loss between model output and actual target)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), target_seq.view(-1))

        # Perplexity is e^(cross-entropy loss)
        perplexity = torch.exp(loss)

    return perplexity.item()


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained LSTM model.")
    parser.add_argument('--model_path', type=str, default='./best_model.pth', help='Path to the trained model file.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing LaTeX data files.')
    parser.add_argument('--length', type=int, default=100, help='The length of the generated text.')
    parser.add_argument('--temperature', type=float, default=0.5, help='The temperature for sampling.')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension of the LSTM model.')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of the LSTM model.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the LSTM model.')
    parser.add_argument('--reference_text', type=str, help='Reference text for BLEU score evaluation.')

    args = parser.parse_args()

    # List of prompts to iterate over
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

    # Load dataset to get vocab size and mappings
    filepaths = [os.path.join(args.data_dir, fname) for fname in os.listdir(args.data_dir) if fname.endswith('.tex')]
    dataset = LaTeXDataset(filepaths)

    # Load the trained model
    model = load_model(args.model_path, dataset.vocab_size, args.embedding_dim, args.hidden_dim, args.num_layers)

    for prompt in prompts:
        print(f"\n=== Generating text for prompt: '{prompt}' ===")

        # Generate text for the current prompt
        generated_text = generate_text(model, dataset, prompt, args.length, args.temperature)
        print(f"\nGenerated Text:\n{generated_text}")

        # Calculate BLEU score if reference text is provided
        if args.reference_text:
            with open(args.reference_text, 'r') as ref_file:
                reference_text = ref_file.read().strip()
                # Show both generated and reference text
                print("\nReference Text:\n", reference_text)
                # Calculate BLEU score
                bleu_score = calculate_bleu(generated_text, [reference_text])
                print(f"\nBLEU Score for prompt '{prompt}': {bleu_score:.4f}")

        # Calculate Perplexity
        perplexity = calculate_perplexity(model, dataset, generated_text)
        print(f"Perplexity for prompt '{prompt}': {perplexity:.4f}")

if __name__ == "__main__":
    main()
