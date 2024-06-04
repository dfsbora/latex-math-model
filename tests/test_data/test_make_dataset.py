import unittest
import os
from data.make_dataset import prepare_data_loader_with_sequences


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.seq_length = 15
        self.test_file_path = "test_data.txt"
        if not os.path.exists(self.test_file_path):
            with open(self.test_file_path, "w") as f:
                f.write("This is a test file for data loader function. eos It contains some sample text for testing, with one very short sentence, and one very long.")
        self.expected_num_batches = 1
        self.expected_num_sequences = 3


    def tearDown(self):
        # Clean up: Delete the test file after the test
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_prepare_data_loader_with_sequences(self):
        dataloader, _ = prepare_data_loader_with_sequences(self.test_file_path, self.batch_size, self.seq_length)

        self.assertIsNotNone(dataloader)

        self.assertEqual(len(dataloader), self.expected_num_batches)

        for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
            self.assertEqual(len(input_batch), self.expected_num_sequences)

            for i in range(len(input_batch)):
                input_data_point = input_batch[i]
                target_data_point = target_batch[i]
                self.assertEqual(len(input_data_point), self.seq_length)
                self.assertEqual(input_data_point[1], target_data_point[0])
                break

if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()
