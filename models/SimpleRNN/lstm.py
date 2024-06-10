import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)


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
        return (weight.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size),
                weight.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size))


vocab_size = dataset.vocab_size
embedding_dim = 256
hidden_dim = 512
num_layers = 2
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers)


# Training Loop with Early Stopping
def train(model, train_loader, val_loader, num_epochs, learning_rate, print_every=100, patience=5):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        hidden = model.init_hidden(train_loader.batch_size)
        hidden = tuple([each.data for each in hidden])
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            hidden = tuple([each.data for each in hidden])  # Detach hidden state
            output, hidden = model(inputs, hidden)
            loss = criterion(output.view(-1, model.vocab_size), targets.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % print_every == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Step {batch_idx}/{len(train_loader)}, Loss: {loss.item()}')

        avg_train_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break


def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        hidden = model.init_hidden(val_loader.batch_size)
        hidden = tuple([each.data for each in hidden])

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output, hidden = model(inputs, hidden)
            loss = criterion(output.view(-1, model.vocab_size), targets.view(-1))
            running_loss += loss.item()

    avg_val_loss = running_loss / len(val_loader)
    return avg_val_loss


num_epochs = 50
learning_rate = 0.002
patience = 5

train(model, train_loader, val_loader, num_epochs, learning_rate, patience=patience)


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
        probs = nn.functional.softmax(output[-1], dim=-1).data.cpu()
        char_idx = torch.multinomial(probs, 1).item()
        chars.append(dataset.idx_to_char[char_idx])
        input_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)

    return ''.join(chars)


start_seq = r"\begin{theorem}"
generated_text = generate_text(model, start_seq, 500, temperature=0.5)
print(generated_text)
