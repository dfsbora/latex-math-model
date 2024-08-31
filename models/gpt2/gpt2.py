import os
import torch
import wandb
import argparse
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from models.utils.evaluate_metrics import (
    calculate_perplexity,
    calculate_bleu_score,
    calculate_rouge_score,
    calculate_token_accuracy,
    calculate_f1_score,
    measure_inference_speed_gpt2,
    compile_latex,
    log_metrics
)

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
        next_token = torch.multinomial(probs, num_samples=1)

        next_token = next_token.squeeze(-1)
        generated = torch.cat((generated, next_token), dim=1)

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text.split()  # Return the generated text as a list of tokens


# Callback for logging with wandb and custom metrics
class CustomWandbCallback(TrainerCallback):
    def __init__(self, model, tokenizer, val_dataset):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        start_seq = r"\begin{theorem}"
        generated_text_tokens = generate_text(self.model, self.tokenizer, start_seq, 500)

        eval_loss = None
        for log in reversed(state.log_history):
            if "eval_loss" in log:
                eval_loss = log["eval_loss"]
                break

        if eval_loss is not None:
            sample_idx = 0
            reference_text = self.tokenizer.decode(self.val_dataset[sample_idx]['labels'], skip_special_tokens=True)
            reference_text_tokens = reference_text.split()

            perplexity = calculate_perplexity(eval_loss)
            bleu_score = calculate_bleu_score(reference_text_tokens, generated_text_tokens)
            rouge_score = calculate_rouge_score(reference_text, ' '.join(generated_text_tokens))
            token_accuracy = calculate_token_accuracy(
                torch.tensor(self.val_dataset[sample_idx]['labels']),
                torch.tensor(self.tokenizer.encode(' '.join(generated_text_tokens), truncation=True, max_length=self.val_dataset[sample_idx]['input_ids'].shape[0]))
            )
            f1_score = calculate_f1_score(
                torch.tensor(self.val_dataset[sample_idx]['labels']).numpy(),
                torch.tensor(self.tokenizer.encode(' '.join(generated_text_tokens), truncation=True, max_length=self.val_dataset[sample_idx]['input_ids'].shape[0])).numpy()
            )
            inference_time = measure_inference_speed_gpt2(self.model, self.tokenizer, start_seq, device=args.device)

            compilation_output, error_count, warning_count = compile_latex(' '.join(generated_text_tokens))

            log_metrics(state.epoch, state.log_history[-1]["loss"], eval_loss, perplexity, bleu_score, rouge_score, token_accuracy, f1_score, inference_time)
            wandb.log({
                "latex_error_count": error_count,
                "latex_warning_count": warning_count
            })
            print(f'Compilation Output:\n{compilation_output}')
            print(f'Errors: {error_count}, Warnings: {warning_count}')
        else:
            print("No eval_loss found in log history.")

        wandb.log({"sampled_text": wandb.Html(f"<pre>{' '.join(generated_text_tokens)}</pre>")})

def main():
    # Argument parser to handle the resume_from_checkpoint parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the checkpoint to resume from")
    args = parser.parse_args()

    # Initialize wandb for tracking experiments
    wandb.init(project="math_latex_project")

    # Directory containing LaTeX data files
    data_dir = "data"
    filepaths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.tex')]
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
        logging_steps=50,
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
    generated_text = generate_text(model, tokenizer, start_seq, 500, temperature=0.5)
    print(generated_text)
    wandb.log({"final_generated_text": wandb.Html(f"<pre>{generated_text}</pre>")})

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
