import os
import unittest

import torch

from models.SimpleRNN.lstm import LaTeXDataset, LSTMModel, generate_text, compile_latex


class TestLaTeXDataset(unittest.TestCase):
    def setUp(self):
        # Setup a small LaTeX dataset for testing
        self.test_dir = os.path.dirname(__file__)
        self.filepath = os.path.join(self.test_dir, 'test.tex')
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write("This is a test dataset for LaTeX. \\begin{equation} E = mc^2 \\end{equation}")

        self.dataset = LaTeXDataset([self.filepath], seq_length=10)

    def tearDown(self):
        # Clean up the test file
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def test_len(self):
        self.assertEqual(len(self.dataset), len(self.dataset.data) - self.dataset.seq_length)

    def test_getitem(self):
        x, y = self.dataset[0]
        self.assertEqual(len(x), self.dataset.seq_length)
        self.assertEqual(len(y), self.dataset.seq_length)


class TestLSTMModel(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 50
        self.embedding_dim = 10
        self.hidden_dim = 20
        self.num_layers = 2
        self.model = LSTMModel(self.vocab_size, self.embedding_dim, self.hidden_dim, self.num_layers)

    def test_forward(self):
        batch_size = 5
        seq_length = 7
        x = torch.randint(0, self.vocab_size, (batch_size, seq_length))
        hidden = self.model.init_hidden(batch_size)
        out, hidden = self.model(x, hidden)
        self.assertEqual(out.shape, (batch_size, seq_length, self.vocab_size))
        self.assertEqual(len(hidden), 2)
        self.assertEqual(hidden[0].shape, (self.num_layers, batch_size, self.hidden_dim))
        self.assertEqual(hidden[1].shape, (self.num_layers, batch_size, self.hidden_dim))


class TestGenerateText(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(__file__)
        self.vocab_size = 256
        self.embedding_dim = 10
        self.hidden_dim = 20
        self.num_layers = 2
        self.model = LSTMModel(self.vocab_size, self.embedding_dim, self.hidden_dim, self.num_layers)
        self.dataset = LaTeXDataset([os.path.join(self.test_dir, 'test.tex')])
        self.dataset.char_to_idx = {chr(i): i for i in range(self.vocab_size)}
        self.dataset.idx_to_char = {i: chr(i) for i in range(self.vocab_size)}

        # Create a dummy 'test.tex' file
        self.filepath = os.path.join(self.test_dir, 'test.tex')
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write("This is a test dataset for LaTeX. \\begin{equation} E = mc^2 \\end{equation}")

    def tearDown(self):
        # Clean up the test file
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def test_generate_text(self):
        start_seq = "test"
        length = 20
        generated_text = generate_text(self.model, self.dataset, start_seq, length, temperature=1.0)
        self.assertEqual(len(generated_text), len(start_seq) + length)


class TestCompileLatex(unittest.TestCase):
    def test_compile_latex(self):
        latex_content = r"""
$$\xymatrix{
\Spec(K) \ar[r] \ar[d] & X \ar[d] \\
\Spec(A) \ar[r] & Y
}
$$
"""
        output, error_count, warning_count = compile_latex(latex_content)
        self.assertIn("Output written on", output)
        self.assertEqual(error_count, 0)
        self.assertEqual(warning_count, 0)


if __name__ == '__main__':
    unittest.main()
