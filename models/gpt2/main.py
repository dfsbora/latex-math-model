import os
import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
import wandb


# Initialize wandb for tracking experiments
wandb.init(project="math_latex_project")


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


# Function for generating text
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


# Callback for logging with wandb
class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % args.logging_steps == 0:
            sample_text = generate_text(model, tokenizer, r"\begin{theorem}", 500)
            wandb.log({"sampled_text": wandb.Html(sample_text)})


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

# Initialize Trainer with EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[WandbCallback()]
)

# Start training
trainer.train()

# Save the best model
trainer.save_model("./results/final_model")
tokenizer.save_pretrained("./results/final_model")


# Generate example text
start_seq = r"\begin{theorem}"
generated_text = generate_text(model, tokenizer, start_seq, 500, temperature=0.5)
print(generated_text)
wandb.log({"final_generated_text": wandb.Html(f"<pre>{generated_text}</pre>")})
