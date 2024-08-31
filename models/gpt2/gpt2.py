import os
import torch
import wandb
import argparse
import numpy as np
import time
import tempfile
import subprocess
import re
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from transformers.trainer_utils import EvalPrediction
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import random

class LaTeXDataset(Dataset):
    def __init__(self, filepaths, tokenizer, seq_length=128):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.examples = self.load_and_tokenize_data(filepaths)

    def load_and_tokenize_data(self, filepaths):
        data = ""
        for filepath in filepaths:
            with open(filepath, 'r', encoding='utf-8') as f:
                data += f.read()
        tokens = self.tokenizer.encode(data, return_tensors="pt", truncation=False, padding=False)
        num_chunks = (tokens.size(1) + self.seq_length - 1) // self.seq_length  # Calculate number of chunks
        input_ids = tokens[0].new_zeros((num_chunks * self.seq_length,))  # Initialize with padding tokens
        input_ids[:tokens.size(1)] = tokens[0]  # Copy tokens to the new tensor
        input_ids = input_ids.view(num_chunks, self.seq_length)
        labels = input_ids.clone()  # Shifted labels for training
        return [{'input_ids': input_ids[i], 'labels': labels[i]} for i in range(len(input_ids))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def generate_text(model, tokenizer, start_seq, length=100, temperature=0.5, top_k=50):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    generated = tokenizer.encode(start_seq, return_tensors="pt").to(device)

    for _ in range(length):
        outputs = model(generated)
        logits = outputs.logits[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # Shape: [batch_size, 1]

        # Ensure next_token has the correct dimensions
        next_token = next_token.squeeze(-1)  # Remove the last dimension if it's 1
        next_token = next_token.unsqueeze(0) if next_token.dim() == 1 else next_token  # Ensure it has the correct batch dimension

        # Concatenate along the sequence length dimension
        generated = torch.cat((generated, next_token), dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)

def compile_latex(latex_content):
    latex_template = r"""
    \documentclass{article}
    \usepackage{amsmath}
    \usepackage{amsthm}
    \usepackage{amsfonts}
    \usepackage{graphicx}
    \usepackage{hyperref}

    \begin{document}

    %s

    \end{document}
    """

    complete_latex_code = latex_template % latex_content

    with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as temp_file:
        tex_path = temp_file.name
        with open(tex_path, 'w') as f:
            f.write(complete_latex_code)

    result = subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = result.stdout.decode('latin1')
    stderr = result.stderr.decode('latin1')

    os.remove(tex_path)
    for ext in ['.aux', '.log', '.pdf']:
        path = tex_path.replace('.tex', ext)
        if os.path.exists(path):
            os.remove(path)

    error_count = len(re.findall(r'! LaTeX Error:', stderr)) + len(re.findall(r'! LaTeX Error:', stdout))
    warning_count = len(re.findall(r'LaTeX Warning:', stderr)) + len(re.findall(r'LaTeX Warning:', stdout))

    return stderr if stderr else stdout, error_count, warning_count

# Custom callback to log BLEU, perplexity, inference speed, and LaTeX errors/warnings to wandb
class CustomWandbCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_dataset):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.logging_counter = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % args.logging_steps == 0:
            start_time = time.time()
            sample_text = generate_text(self.model, self.tokenizer, r"\begin{theorem}", 500)
            end_time = time.time()

            # Calculate and log inference speed (time taken for generation)
            inference_time = end_time - start_time
            wandb.log({
                "sampled_text": wandb.Html(sample_text),
                "inference_time": inference_time
            })

            # Increment logging counter
            self.logging_counter += 1

            # Log LaTeX errors and warnings every 10th logging interval
            if self.logging_counter % 10 == 0:
                _, error_count, warning_count = compile_latex(sample_text)
                wandb.log({
                    "latex_error_count": error_count,
                    "latex_warning_count": warning_count
                })


    def on_evaluate(self, args, state, control, metrics, **kwargs):
        num_samples = min(2000, len(self.eval_dataset)) #eval on random 2k samples
        eval_indices = random.sample(range(len(self.eval_dataset)), num_samples)
        eval_subset = [self.eval_dataset[i] for i in eval_indices]

        # Decode references
        references = [self.tokenizer.decode(item['labels'], skip_special_tokens=True) for item in eval_subset]

        # Generate predictions using your generate_text function
        predictions = []
        for i, ref in enumerate(references):
            pred = generate_text(self.model, self.tokenizer, ref[:50], length=50, temperature=0.5, top_k=50)
            predictions.append(pred.split())

            # Log progress every 100 sentences
            if i % 100 == 0:
                print(f"Generated {i+1}/{len(references)} predictions.")

        # BLEU score calculation
        list_of_references = [[ref.split()] for ref in references]  # Need list of lists for corpus_bleu
        avg_bleu_score = corpus_bleu(list_of_references, predictions)

        # Calculate perplexity
        eval_loss = metrics["eval_loss"]
        perplexity = torch.exp(torch.tensor(eval_loss))

        # Log to WandB
        wandb.log({"eval_bleu": avg_bleu_score, "eval_perplexity": perplexity.item()})
        print("Evaluation completed and logged.")



def main():
    # Argument parser to handle the resume_from_checkpoint parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the checkpoint to resume from")
    args = parser.parse_args()

    # Initialize wandb for tracking experiments
    wandb.init(project="math_latex_project")

    # Define hyperparameters using wandb.config
    wandb.config.temperature = 0.5  # Default temperature value

    # Directory containing LaTeX data files
    data_dir = "data"
    #filepaths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.tex')]
    filepaths = [os.path.join(data_dir, "data.tex")]
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    # Initialize dataset
    dataset = LaTeXDataset(filepaths, tokenizer)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Initialize model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    # Data collator for handling token batching
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments setup
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        prediction_loss_only=False,
        report_to="wandb",
    )

    # Initialize Trainer with CustomWandbCallback
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[CustomWandbCallback(model, tokenizer, val_dataset)]
    )

    # Resume from checkpoint if provided
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # Save the best model
    trainer.save_model("./results/final_model")
    tokenizer.save_pretrained("./results/final_model")

    # Generate example text
    start_seq = r"\begin{theorem}"
    generated_text = generate_text(model, tokenizer, start_seq, 500)
    print(generated_text)
    wandb.log({"final_generated_text": wandb.Html(f"<pre>{generated_text}</pre>")})

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
