import unittest
import torch
from models.lstm import LSTMModel

class TestLSTMModel(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.embedding_dim = 10
        self.hidden_dim = 20
        self.num_layers = 2
        
        # Initialize the model
        self.model = LSTMModel(self.vocab_size, self.embedding_dim, self.hidden_dim, self.num_layers)
        self.model.eval()

    def test_model_output_shape(self):
        """Test if the output of the model has the correct shape."""
        batch_size = 5
        seq_length = 10
        test_input = torch.randint(0, self.vocab_size, (batch_size, seq_length))
        hidden = self.model.init_hidden(batch_size)
        
        outputs, hidden = self.model(test_input, hidden)
        
        expected_shape = (batch_size * seq_length, self.vocab_size)
        self.assertEqual(outputs.shape, expected_shape, f"Expected output shape to be {expected_shape}, got {outputs.shape}")

    def test_hidden_state_shape(self):
        """Test if the hidden state has the correct shape."""
        batch_size = 3
        seq_length = 7
        test_input = torch.randint(0, self.vocab_size, (batch_size, seq_length))
        hidden = self.model.init_hidden(batch_size)
        
        self.assertEqual(hidden[0].shape, (self.num_layers, batch_size, self.hidden_dim))
        self.assertEqual(hidden[1].shape, (self.num_layers, batch_size, self.hidden_dim))
        
        _, new_hidden = self.model(test_input, hidden)
        self.assertEqual(new_hidden[0].shape, (self.num_layers, batch_size, self.hidden_dim))
        self.assertEqual(new_hidden[1].shape, (self.num_layers, batch_size, self.hidden_dim))

    def test_init_hidden_zeros(self):
        """Check if the initial hidden states are zero tensors."""
        batch_size = 4
        hidden = self.model.init_hidden(batch_size)

        self.assertTrue(torch.all(hidden[0] == 0))
        self.assertTrue(torch.all(hidden[1] == 0))

if __name__ == '__main__':
    unittest.main()