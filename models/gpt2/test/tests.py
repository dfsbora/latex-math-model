import os
import unittest

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from models.gpt2.gpt2 import LaTeXDataset, generate_text


class TestLaTeXDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(__file__)
        self.filepath = os.path.join(self.test_dir, 'test.tex')
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write("This is a test dataset for LaTeX. \\begin{equation} E = mc^2 \\end{equation} test")

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.dataset = LaTeXDataset([self.filepath], self.tokenizer, seq_length=10)

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def test_len(self):
        text = "This is a test dataset for LaTeX. \\begin{equation} E = mc^2 \\end{equation}"
        expected_length = (len(self.tokenizer.encode(text)) + 9) // 10
        self.assertEqual(len(self.dataset), expected_length)

    def test_getitem(self):
        item = self.dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('labels', item)
        self.assertEqual(len(item['input_ids']), 10)
        self.assertEqual(len(item['labels']), 10)


class TestGenerateText(unittest.TestCase):
    def setUp(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    def test_generate_text(self):
        start_seq = "test"
        length = 20
        generated_text = generate_text(self.model, self.tokenizer, start_seq, length, temperature=1.0)
        self.assertTrue(len(generated_text) > len(start_seq))


if __name__ == '__main__':
    unittest.main()
