import math
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2Tokenizer
import wandb
from tqdm import tqdm
import re

# Initialize Weights & Biases
wandb.init(project="math_latex_project")

class LatexDataset(Dataset):
    def __init__(self, directory, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id  # Store pad_token_id for reference
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tex')]
        self.data = self.load_data()

    def preprocess_latex(self, text):
        # Custom preprocessing for LaTeX content
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

# Load the tokenizer from the fine-tuned model directory
tokenizer = GPT2Tokenizer.from_pretrained('./results/final_model')
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '', 'eos_token': ''})

# Prepare dataset
dataset = LatexDataset("data", tokenizer)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################### Custom Transformer Model with Masked Attention ######################

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 max_seq_length, pad_token_id):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        self.encoder = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
        self.decoder = nn.Linear(d_model, vocab_size)

    #     self._reset_parameters()
    #
    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt):
        # Assuming src and tgt are [batch_size, seq_len, embedding_dim]
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)

        # Generate target mask and source mask
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)

        # Create key padding masks ensuring they are 2D: [batch_size, seq_len]
        src_padding_mask = (src == self.pad_token_id).to(src.device)
        tgt_padding_mask = (tgt == self.pad_token_id).to(tgt.device)

        # Reduce any excess dimensions
        if src_padding_mask.dim() > 2:
            src_padding_mask = src_padding_mask.squeeze(-1)
        if tgt_padding_mask.dim() > 2:
            tgt_padding_mask = tgt_padding_mask.squeeze(-1)

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src = self.encoder(src) * math.sqrt(self.d_model)  # embedding the source
        tgt = self.encoder(tgt) * math.sqrt(self.d_model)  # embedding the target

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        src = src.transpose(0, 1)  # should be [seq_len, batch_size, d_model]
        tgt = tgt.transpose(0, 1)  # should be [seq_len, batch_size, d_model]

        if src.shape[-1] != self.d_model or tgt.shape[-1] != self.d_model:
            raise RuntimeError("The feature number of src and tgt must be equal to d_model")

        output = self.transformer(src, tgt, src_mask, tgt_mask, None,
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask)

        output = output.transpose(0, 1)  # back to [batch_size, seq_len, d_model]
        output = self.decoder(output)
        return output

    def generate_text(self, tokenizer, prompt=None, max_length=512, repetition_penalty=1.2):
        self.eval()
        with torch.no_grad():
            if prompt is None:
                sample_text = dataset[0]['input_ids'].squeeze(0).to(device)[:100]  # Sample first 100 tokens
            else:
                tokenized_prompt = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_length, padding=True)
                sample_text = tokenized_prompt['input_ids'].squeeze(0).to(device)

            generated = sample_text.unsqueeze(0)

            for _ in range(max_length - len(sample_text)):
                tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(device)
                src_padding_mask = (generated == self.pad_token_id).type(torch.bool).to(device)
                tgt_padding_mask = (generated == self.pad_token_id).type(torch.bool).to(device)

                outputs = self(generated, generated, tgt_mask=tgt_mask,
                               src_key_padding_mask=src_padding_mask,
                               tgt_key_padding_mask=tgt_padding_mask)
                logits = outputs[:, -1, :]
                logits = self.apply_repetition_penalty(logits, generated, repetition_penalty)
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token.item() == tokenizer.eos_token_id:  # Stop if end of sequence token is generated
                    break

            generated_text = tokenizer.decode(generated.squeeze().tolist(), skip_special_tokens=True)
            return generated_text

    @staticmethod
    def apply_repetition_penalty(logits, generated, repetition_penalty):
        for i in range(generated.size(1)):
            logits[:, generated[0, i]] /= repetition_penalty
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


###################### Training Loop ######################

# Define model parameters
vocab_size = len(tokenizer)  # Adjust vocab_size to include special tokens
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
max_seq_length = 512
pad_token_id = tokenizer.pad_token_id

# Initialize the custom transformer model
student_model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, pad_token_id).to(device)

# Load the previously trained student model
student_model.load_state_dict(torch.load('./checkpoints/final_model.pt'))

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id).to(device)
optimizer = AdamW(student_model.parameters(), lr=0.0001)

# Define checkpoint directory
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
num_epochs = 10
sample_interval = 2  # Sample generated text every 2 epochs

for epoch in range(num_epochs):
    student_model.train()
    running_loss = 0.0
    batch_count = 0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        labels = inputs.clone()

        optimizer.zero_grad()

        # Forward pass through the student model
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = student_model.create_mask(inputs, inputs)
        student_outputs = student_model(inputs, inputs, src_mask=src_mask, tgt_mask=tgt_mask,
                                        src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        student_logits = student_outputs.view(-1, vocab_size)

        # Compute loss
        loss = criterion(student_logits, labels.view(-1))
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        batch_count += 1

        # Log loss for each batch
        print(f"Batch {batch_count}, Loss: {loss.item()}")
        wandb.log({"batch_loss": loss.item()})

    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_loss}")
    wandb.log({"epoch": epoch + 1, "training_loss": avg_loss})

    # Validation
    student_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Validation {epoch + 1}/{num_epochs}"):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            labels = inputs.clone()

            # Forward pass through the student model
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = student_model.create_mask(inputs, inputs)
            student_outputs = student_model(inputs, inputs, src_mask=src_mask, tgt_mask=tgt_mask,
                                            src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
            student_logits = student_outputs.view(-1, vocab_size)

            # Compute loss
            loss = criterion(student_logits, labels.view(-1))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")
    wandb.log({"epoch": epoch + 1, "validation_loss": avg_val_loss})

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
    torch.save(student_model.state_dict(), checkpoint_path)

    # Sample generated text
    if (epoch + 1) % sample_interval == 0:
        student_model.eval()
        with torch.no_grad():
            prompt = r"\begin{theorem}"
            generated_text_samples = student_model.generate_text(tokenizer, prompt=prompt, max_length=512, repetition_penalty=1.2)
            print(f"Sample generated text at epoch {epoch + 1}:\n{generated_text_samples}")
            wandb.log({"sample_text": wandb.Html(f"<pre>{generated_text_samples}</pre>")})

# Save the final model
final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
torch.save(student_model.state_dict(), final_model_path)
print("Training completed. Final model saved.")
