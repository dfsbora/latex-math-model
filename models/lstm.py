import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # Final dense layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        # Embedding input words
        embeds = self.embedding(x)
        
        # Passing the embeddings through LSTM
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # Reshape the output from [batch_size, sequence_length, hidden_dim]
        # to [batch_size * sequence_length, hidden_dim]
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # Getting the final output from the dense layer
        output = self.fc(lstm_out)
        
        return output, hidden

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes num_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
        return hidden