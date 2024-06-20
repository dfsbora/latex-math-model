import unittest
import os
from data.simplify_dataset import mask_formulas, mask_text

class TestMaskFormulas(unittest.TestCase):
    def test_single_dollar_formulas(self):
        input_text = "Here is a formula $E=mc^2$ in the text."
        input_masking_word = "FORMULA"
        expected_output = "Here is a formula $FORMULA$ in the text."
        result = mask_formulas(input_text, input_masking_word)
        self.assertEqual(result, expected_output)

    def test_double_dollar_formulas(self):
        input_text = """Here is a formula
        $$
        E=mc^2
        $$
        in the text.
        """
        input_masking_word = "FORM"
        expected_output = """Here is a formula
        $$FORM$$
        in the text.
        """
        result = mask_formulas(input_text, input_masking_word)
        self.assertEqual(result, expected_output)

    def test_mixed_formulas(self):
        input_text = """Here is a single $E=mc^2$ and a double
        $$
        E=mc^2
        $$
        formula.
        """
        input_masking_word = "FORMULA"
        expected_output = """Here is a single $FORMULA$ and a double
        $$FORMULA$$
        formula.
        """
        result = mask_formulas(input_text, input_masking_word)
        self.assertEqual(result, expected_output)

    def test_no_formulas(self):
        input_text = "Here is a text without any formulas."
        input_masking_word = "FORMULA"
        expected_output = input_text
        result = mask_formulas(input_text, input_masking_word)
        self.assertEqual(result, expected_output)

    def test_escaped_dollars(self):
        input_text = "Here is a text with escaped dollars \\$E=mc^2\\$."
        input_masking_word = "FORMULA"
        expected_output = input_text
        result = mask_formulas(input_text, input_masking_word)
        self.assertEqual(result, expected_output)

    def test_multiple_formulas(self):
        input_text = "Multiple formulas: $E=mc^2$ and $a^2 + b^2 = c^2$."
        input_masking_word = "FORMULA"
        expected_output = "Multiple formulas: $FORMULA$ and $FORMULA$."
        result = mask_formulas(input_text, input_masking_word)
        self.assertEqual(result, expected_output)


class TestMaskText(unittest.TestCase):

    def test_single_dollar_formulas(self):
        input_text = "Here is a formula $E=mc^2$ in the text."
        input_masking_word = "TEXT"
        expected_output = "TEXT $E=mc^2$ TEXT"
        result = mask_text(input_text, input_masking_word)
        self.assertEqual(result, expected_output)

    def test_double_dollar_formulas(self):
        input_text = """Here is a formula
        $$
        E=mc^2
        $$"""
        input_masking_word = "TEXT"
        expected_output = """TEXT
        $$
        E=mc^2
        $$"""
        result = mask_text(input_text, input_masking_word)
        self.assertEqual(result, expected_output)

    def test_mixed_formulas(self):
        input_text = """Here is a single $E=mc^2$ and a double
        $$
        E=mc^2
        $$
        formula."""
        input_masking_word = "TEXTO"
        expected_output = """TEXTO $E=mc^2$ TEXTO
        $$
        E=mc^2
        $$
TEXTO"""
        result = mask_text(input_text, input_masking_word)
        self.assertEqual(result, expected_output)

    def test_no_formulas(self):
        input_text = "Here is a text without any formulas."
        input_masking_word = "TEXT"
        expected_output = "TEXT"
        result = mask_text(input_text, input_masking_word)
        self.assertEqual(result, expected_output)

    def test_multiple_formulas(self):
        input_text = "Multiple formulas: $E=mc^2$ and $a^2 + b^2 = c^2$."
        input_masking_word = "TEXT"
        expected_output = "TEXT $E=mc^2$ TEXT $a^2 + b^2 = c^2$ TEXT"
        result = mask_text(input_text, input_masking_word)
        self.assertEqual(result, expected_output)


    def test_citation(self):
        input_text = "Here is a formula $E=mc^2$ and a citation \cite{example}."
        input_masking_word = "TEXT"
        expected_output = "TEXT $E=mc^2$ TEXT \cite{example} TEXT"
        result = mask_text(input_text, input_masking_word)
        self.assertEqual(result, expected_output)

    def test_citation_begin(self):
        input_text = "\cite{example} and a formula $E=mc^2$."
        input_masking_word = "TEXT"
        expected_output = " \cite{example} TEXT $E=mc^2$ TEXT"
        result = mask_text(input_text, input_masking_word)
        self.assertEqual(result, expected_output)

    def test_item(self):
        input_text = """\\begin{enumerate}
\item Jarod Alper contributed a chapter discussing the literature"""
        input_masking_word = "TEXT"
        expected_output = """ \\begin{enumerate}
\item TEXT"""
        result = mask_text(input_text, input_masking_word)
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()