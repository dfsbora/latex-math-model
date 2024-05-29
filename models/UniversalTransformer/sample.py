"""Samples from the trained Universal Transformer model.

The sampling code includes a function to generate text using the model,
leveraging techniques such as greedy decoding or beam search for generating the output sequence.
"""

from models.UniversalTransformer.LaTeXTokenizer import LaTeXTokenizer
from models.UniversalTransformer.UniversalTransformer import UniversalTransformer
from models.UniversalTransformer.train import load_model
from models.UniversalTransformer.utils import generate_text


def main():
    data_dir = "data"
    tokenizer_path = "tokenizer.json"
    # TODO: use the best saved model for sampling
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
        model, tokenizer.tokenizer, start_sequence, max_length=256, temperature=1.0, top_k=0, top_p=0.0)
    print("Generated Text:\n", generated_text)


if __name__ == "__main__":
    main()
