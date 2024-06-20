import unittest
import os
import tempfile
from data.data_experiment import LatexFileParser, LatexFileEOSCounter




class TestLatexFileProcessor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_file_path = os.path.join(self.test_dir.name, 'test_file.tex')

        with open(self.test_file_path, 'w') as f:
            f.write(r"""
                \documentclass{article}
                \begin{document}
                This is a sample document with some formulas $a = b + c$ and a label \label{sec:intro}.
                \section{Introduction}
                Here is a citation \cite{author2021}.
                $$
                E = mc^2
                $$
                \end{document}
            """)

        self.processor = LatexFileParser(self.test_file_path)

    def tearDown(self):
        # Cleanup the temporary directory
        self.test_dir.cleanup()

    def test_make_list_formulas(self):
        output_file = os.path.join(self.test_dir.name, 'formulas.txt')
        self.processor.make_list_formulas(output_file)

        with open(output_file, 'r') as f:
            content = f.read()

        self.assertIn('a = b + c', content)
        self.assertIn('E = mc^2', content)

    def test_make_list_labels(self):
        output_file = os.path.join(self.test_dir.name, 'labels.txt')
        self.processor.make_list_labels(output_file)

        with open(output_file, 'r') as f:
            content = f.read()

        self.assertIn('sec:intro', content)

    def test_make_list_cites(self):
        output_file = os.path.join(self.test_dir.name, 'cites.txt')
        self.processor.make_list_cites(output_file)

        with open(output_file, 'r') as f:
            content = f.read()

        self.assertIn('author2021', content)



if __name__ == '__main__':
    unittest.main()
