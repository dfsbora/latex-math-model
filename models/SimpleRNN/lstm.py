import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import wandb
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import subprocess
import tempfile
import re

# Initialize wandb
wandb.init(project="math_latex_project")

# Data Preparation and Preprocessing
class LaTeXDataset(Dataset):
    def __init__(self, filepaths, seq_length=100):
        self.data = self.load_data(filepaths)
        self.chars = sorted(list(set(self.data)))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        self.seq_length = seq_length
        self.vocab_size = len(self.chars)

    def load_data(self, filepaths):
        data = ""
        for filepath in filepaths:
            with open(filepath, 'r', encoding='utf-8') as f:
                data += f.read()
        return data

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x_str = self.data[idx:idx + self.seq_length]
        y_str = self.data[idx + 1:idx + self.seq_length + 1]
        x = torch.tensor([self.char_to_idx[char] for char in x_str], dtype=torch.long)
        y = torch.tensor([self.char_to_idx[char] for char in y_str], dtype=torch.long)
        return x, y


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets


# Load data
data_dir = "data"  # Path to the directory containing LaTeX data
filepaths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.tex')]
dataset = LaTeXDataset(filepaths)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)


# Text Generation
def generate_text(model, start_seq, length, temperature=1.0):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    chars = [char for char in start_seq]
    input_seq = torch.tensor([dataset.char_to_idx[char] for char in chars], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    hidden = tuple([each.data for each in hidden])

    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        output = output / temperature
        probs = nn.functional.softmax(output[0, -1], dim=-1).data.cpu()
        char_idx = torch.multinomial(probs, 1).item()
        chars.append(dataset.idx_to_char[char_idx])
        input_seq = torch.cat((input_seq, torch.tensor([[char_idx]], dtype=torch.long).to(device), dim=1)

    return ''.join(chars)


# Model Definition
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(weight.device),
                weight.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(weight.device))


vocab_size = dataset.vocab_size
embedding_dim = 256  # Bigger embedding dimension
hidden_dim = 512  # Bigger hidden dimension
num_layers = 2  # More layers
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)


# Helper function to compile LaTeX and return errors
def compile_latex(latex_content):
    # Define the template
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

    # Insert the generated content into the template
    complete_latex_code = latex_template % latex_content

    with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as temp_file:
        tex_path = temp_file.name
        with open(tex_path, 'w') as f:
            f.write(complete_latex_code)

    result = subprocess.run(['pdflatex', '-interaction=nonstopmode', tex_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = result.stdout.decode('utf-8')
    stderr = result.stderr.decode('utf-8')

    # Clean up the generated files
    os.remove(tex_path)
    aux_path = tex_path.replace('.tex', '.aux')
    log_path = tex_path.replace('.tex', '.log')
    pdf_path = tex_path.replace('.tex', '.pdf')
    for path in [aux_path, log_path, pdf_path]:
        if os.path.exists(path):
            os.remove(path)

    # Count errors and warnings
    error_count = 0
    warning_count = 0

    error_count += len(re.findall(r'! LaTeX Error:', stderr)) + len(re.findall(r'! LaTeX Error:', stdout))
    warning_count += len(re.findall(r'LaTeX Warning:', stderr)) + len(re.findall(r'LaTeX Warning:', stdout))

    return stderr if stderr else stdout, error_count, warning_count


# Training Loop with Early Stopping and wandb Logging
def train(model, train_loader, val_loader, num_epochs, learning_rate, print_every=1, patience=3, checkpoint_dir='checkpoints'):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()  # Ensure model is in training mode
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size)
            optimizer.zero_grad()
            hidden = tuple([each.data for each in hidden])  # Detach hidden state
            output, hidden = model(inputs, hidden)
            loss = criterion(output.view(-1, model.vocab_size), targets.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_loss, perplexity, bleu_score = evaluate(model, val_loader, criterion, device)

        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}, BLEU Score: {bleu_score:.4f}')

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1, 
            "train_loss": avg_train_loss, 
            "val_loss": val_loss, 
            "perplexity": perplexity, 
            "bleu_score": bleu_score
        })

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')

        # Generate and compile LaTeX for error logging
        start_seq = r"\begin{theorem}"
        sampled_text = generate_text(model, start_seq, 500, temperature=0.5)
        print(f'Sampled Text at Epoch {epoch + 1}:\n{sampled_text}')
        wandb.log({"sampled_text": wandb.Html(sampled_text)})

        # Compile the generated text, count errors, and log them
        compilation_output, error_count, warning_count = compile_latex(sampled_text)
        print(f'Compilation Output at Epoch {epoch + 1}:\n{compilation_output}')
        print(f'Errors: {error_count}, Warnings: {warning_count}')
        
        # Log errors and warnings with consistent key
        wandb.log({
            "latex_error_count": error_count,
            "latex_warning_count": warning_count
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break


# Evaluation Function
def evaluate(model, val_loader, criterion, device):
    model.eval()  # Ensure model is in evaluation mode
    running_loss = 0.0
    total_ppl = 0
    total_bleu = 0
    smoothing = SmoothingFunction().method1

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size)
            hidden = tuple([each.data for each in hidden])
            output, hidden = model(inputs, hidden)
            loss = criterion(output.view(-1, model.vocab_size), targets.view(-1))
            running_loss += loss.item()

            # Perplexity
            log_probs = nn.functional.log_softmax(output, dim=-1)
            batch_ppl = torch.exp(-log_probs)
            total_ppl += batch_ppl.mean().item()

            # BLEU Score
            decoded_preds = torch.argmax(log_probs, dim=-1)
            target_sentences = targets.cpu().numpy().tolist()
            pred_sentences = decoded_preds.cpu().numpy().tolist()

            for target_seq, pred_seq in zip(target_sentences, pred_sentences):
                # Skip padding 0 in targets
                target_seq = [i for i in target_seq if i != 0]
                pred_seq = [i for i in pred_seq if i != 0]

                target_text = [dataset.idx_to_char[idx] for idx in target_seq]
                pred_text = [dataset.idx_to_char[idx] for idx in pred_seq]

                if len(pred_text) > 0 and len(target_text) > 0:
                    total_bleu += sentence_bleu([target_text], pred_text, smoothing_function=smoothing)

    avg_val_loss = running_loss / len(val_loader)
    perplexity = total_ppl / len(val_loader)
    bleu_score = total_bleu / len(val_loader)
    return avg_val_loss, perplexity, bleu_score


num_epochs = 5  # Use fewer epochs for quick verification
learning_rate = 0.002
patience = 3
print_every = 1  # Print and log every epoch

train(model, train_loader, val_loader, num_epochs, learning_rate, patience=patience)


start_seq = r"\begin{theorem}"
generated_text = generate_text(model, start_seq, 500, temperature=0.5)
print(generated_text)
wandb.log({"final_generated_text": wandb.Html(generated_text)})

# Compile the final generated text, count errors, and log them
final_compilation_output, final_error_count, final_warning_count = compile_latex(generated_text)
print(f'Final Compilation Output:\n{final_compilation_output}')
print(f'Final Errors: {final_error_count}, Final Warnings: {final_warning_count}')
wandb.log({
    "final_compilation_output": wandb.Html('<pre>' + final_compilation_output + '</pre>'),
    "final_errors": final_error_count,
    "final_warnings": final_warning_count
})
