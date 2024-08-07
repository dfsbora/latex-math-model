import os
import unittest

import torch
import wandb
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from models.transformer.main_student import LatexDataset, TransformerModel, train


class TestLatexDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        cls.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '', 'eos_token': ''})
        os.makedirs('test_data', exist_ok=True)
        with open('test_data/test.tex', 'w') as f:
            f.write("This is a test dataset for LaTeX.")

    @classmethod
    def tearDownClass(cls):
        os.remove('test_data/test.tex')
        os.rmdir('test_data')

    def test_dataset_loading(self):
        dataset = LatexDataset('test_data', self.tokenizer)
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        self.assertIn('input_ids', sample)
        self.assertIn('attention_mask', sample)


class TestTransformerModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure the directory exists
        cls.test_data_dir = 'test_data'
        os.makedirs(cls.test_data_dir, exist_ok=True)

        # Create a sample test file
        cls.test_file_path = os.path.join(cls.test_data_dir, 'test.tex')
        with open(cls.test_file_path, 'w') as f:
            f.write("This is a test dataset for LaTeX. " * 100)  # Ensure sufficient data

        cls.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        cls.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '', 'eos_token': ''})

        cls.vocab_size = len(cls.tokenizer)
        cls.d_model = 512
        cls.nhead = 8
        cls.num_encoder_layers = 6
        cls.num_decoder_layers = 6
        cls.dim_feedforward = 2048
        cls.max_seq_length = 512
        cls.pad_token_id = cls.tokenizer.pad_token_id
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cls.model = TransformerModel(
            cls.vocab_size, cls.d_model, cls.nhead, cls.num_encoder_layers,
            cls.num_decoder_layers, cls.dim_feedforward, cls.max_seq_length, cls.pad_token_id
        ).to(cls.device)

    @classmethod
    def tearDownClass(cls):
        # Clean up the test file and directory
        if os.path.exists(cls.test_file_path):
            os.remove(cls.test_file_path)
        if os.path.exists(cls.test_data_dir):
            os.rmdir(cls.test_data_dir)

    def test_model_forward(self):
        src = torch.randint(0, self.vocab_size, (2, 512)).to(self.device)
        tgt = torch.randint(0, self.vocab_size, (2, 512)).to(self.device)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.model.create_mask(src, tgt)
        output = self.model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        self.assertEqual(output.shape, (2, 512, self.vocab_size))

    def test_generate_text(self):
        dataset = LatexDataset(self.test_data_dir, self.tokenizer)
        prompt = r"\begin{theorem}"
        generated_text = self.model.generate_text(self.tokenizer, dataset, self.device, prompt=prompt, max_length=20)
        self.assertIsInstance(generated_text, str)
        self.assertGreater(len(generated_text), 0)


class SyntheticLatexDataset(Dataset):
    def __init__(self, tokenizer, pad_token_id, size=1000):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.data = ["This is a test dataset for LaTeX. " * 20] * size
        self.data = [self.tokenizer.encode(text, return_tensors='pt').squeeze(0) for text in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        attention_mask = torch.ones_like(input_ids)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}


class TestTrainingProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize wandb
        wandb.init(project="math_latex_project", mode="disabled")

        cls.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        cls.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '', 'eos_token': ''})
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create synthetic dataset
        cls.dataset = SyntheticLatexDataset(cls.tokenizer, cls.tokenizer.pad_token_id, size=1000)

        train_size = int(0.8 * len(cls.dataset))
        val_size = len(cls.dataset) - train_size

        cls.train_dataset, cls.val_dataset = random_split(cls.dataset, [train_size, val_size])
        cls.train_dataloader = DataLoader(cls.train_dataset, batch_size=2, shuffle=True)
        cls.val_dataloader = DataLoader(cls.val_dataset, batch_size=2, shuffle=False)

        cls.teacher_model = GPT2LMHeadModel.from_pretrained('gpt2')
        cls.teacher_model.eval()
        cls.teacher_model.to(cls.device)

        cls.vocab_size = len(cls.tokenizer)
        cls.d_model = 512
        cls.nhead = 8
        cls.num_encoder_layers = 6
        cls.num_decoder_layers = 6
        cls.dim_feedforward = 2048
        cls.max_seq_length = 512
        cls.pad_token_id = cls.tokenizer.pad_token_id

        cls.student_model = TransformerModel(
            cls.vocab_size, cls.d_model, cls.nhead, cls.num_encoder_layers,
            cls.num_decoder_layers, cls.dim_feedforward, cls.max_seq_length, cls.pad_token_id
        ).to(cls.device)

    @classmethod
    def tearDownClass(cls):
        # Finish wandb run
        wandb.finish()

    def test_training_loop(self):
        train(
            self.student_model, self.teacher_model, self.dataset, self.train_dataloader, self.val_dataloader,
            self.vocab_size, self.tokenizer, self.device, num_epochs=1, sample_interval=1, checkpoint_dir="test_checkpoints"
        )
        self.assertTrue(os.path.exists("test_checkpoints/model_epoch_1.pt"))
        self.assertTrue(os.path.exists("test_checkpoints/final_model.pt"))

        # Clean up checkpoint directory
        os.remove("test_checkpoints/model_epoch_1.pt")
        os.remove("test_checkpoints/final_model.pt")
        os.rmdir("test_checkpoints")


if __name__ == '__main__':
    unittest.main()
