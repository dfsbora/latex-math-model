import unittest
import tempfile
import os
from utils import correct_latex_environments, EvaluatePrompt


class TestEvaluatePrompt(unittest.TestCase):
    def setUp(self):
        # Create temporary files for cites and labels
        self.temp_cites_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8')
        self.temp_labels_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8')

        # Write sample data to cites file
        self.temp_cites_file.write("cite1\ncite2\ncite3\n")
        self.temp_cites_file.close()

        # Write sample data to labels file
        self.temp_labels_file.write("label1\nlabel2\nlabel3\n")
        self.temp_labels_file.close()

        # Sample text
        self.sample_text = r"""
        This is a sample text with some citations and labels.
        Here is one citation: \cite{cite1} and another one: \cite{cite4}.
        Here is a label: \label{label1} and another one: \label{label4}.
        """

        self.evaluator = EvaluatePrompt(
            text=self.sample_text,
            cites_file=self.temp_cites_file.name,
            labels_file=self.temp_labels_file.name
        )

    def tearDown(self):
        # Remove temporary files
        os.remove(self.temp_cites_file.name)
        os.remove(self.temp_labels_file.name)

    def test_initialization(self):
        self.assertEqual(self.evaluator.text, self.sample_text)
        self.assertEqual(self.evaluator.cites_list, {"cite1", "cite2", "cite3"})
        self.assertEqual(self.evaluator.labels_list, {"label1", "label2", "label3"})

    def test_set_text(self):
        new_text = r"""
        Another sample text with different citations and labels.
        Citation: \cite{cite2}. Label: \label{label2}.
        """
        self.evaluator.set_text(new_text)
        self.assertEqual(self.evaluator.text, new_text)

    def test_compare_labels_with_original_labels(self):
        num_generated, num_copy, num_original = self.evaluator.compare_labels_with_original_labels()
        self.assertEqual(num_generated, 2)
        self.assertEqual(num_copy, 1)
        self.assertEqual(num_original, 1)

    def test_compare_cites_with_original_cites(self):
        num_generated, num_copy, num_original = self.evaluator.compare_cites_with_original_cites()
        self.assertEqual(num_generated, 2)
        self.assertEqual(num_copy, 1)
        self.assertEqual(num_original, 1)


if __name__ == "__main__":
    unittest.main()

class TestUtils(unittest.TestCase):
    def test_unopened_environment(self):
        input_text = r"""
given that X equal Y
\end{theorem}
"""

        expected_output = r"""
given that X equal Y
"""
        expected_num_corrected_env = 0
        expected_num_closed_env = 0
        expected_num_deleted_end = 1

        result, num_corrected_env, num_closed_env, num_deleted_end = correct_latex_environments(input_text)
        self.assertEqual(result.strip(), expected_output.strip())
        self.assertEqual(num_corrected_env, expected_num_corrected_env)
        self.assertEqual(num_closed_env, expected_num_closed_env)
        self.assertEqual(num_deleted_end, expected_num_deleted_end)

    def test_unclosed_environment(self):
        input_text = r"""\begin{theorem}
given that X equal Y
"""

        expected_output = r"""\begin{theorem}
given that X equal Y
\end{theorem}
"""
        expected_num_corrected_env = 0
        expected_num_closed_env = 1
        expected_num_deleted_end = 0

        result, num_corrected_env, num_closed_env, num_deleted_end = correct_latex_environments(input_text)
        self.assertEqual(result.strip(), expected_output.strip())
        self.assertEqual(num_corrected_env, expected_num_corrected_env)
        self.assertEqual(num_closed_env, expected_num_closed_env)
        self.assertEqual(num_deleted_end, expected_num_deleted_end)

    def test_no_environment(self):
        input_text = r"""This is a regular text with no environments."""

        expected_output = input_text
        expected_num_corrected_env = 0
        expected_num_closed_env = 0
        expected_num_deleted_end = 0

        result, num_corrected_env, num_closed_env, num_deleted_end = correct_latex_environments(input_text)
        self.assertEqual(result.strip(), expected_output.strip())
        self.assertEqual(num_corrected_env, expected_num_corrected_env)
        self.assertEqual(num_closed_env, expected_num_closed_env)
        self.assertEqual(num_deleted_end, expected_num_deleted_end)

    def test_correct_environments(self):
        input_text = r"""\begin{theorem}
given that X equal Y
\end{lemma}

\begin{proof}
given that the conditions suffices
$$
A = B \cdot C
$$
\end{lemma}

\begin{lemma}
the matrix $a=b$
"""

        expected_output = r"""\begin{theorem}
given that X equal Y
\end{theorem}

\begin{proof}
given that the conditions suffices
$$
A = B \cdot C
$$
\end{proof}

\begin{lemma}
the matrix $a=b$
\end{lemma}
"""

        expected_num_corrected_env = 2
        expected_num_closed_env = 1
        expected_num_deleted_end = 0

        result, num_corrected_env, num_closed_env, num_deleted_end = correct_latex_environments(input_text)
        self.assertEqual(result.strip(), expected_output.strip())
        self.assertEqual(num_corrected_env, expected_num_corrected_env)
        self.assertEqual(num_closed_env, expected_num_closed_env)
        self.assertEqual(num_deleted_end, expected_num_deleted_end)



if __name__ == '__main__':
    unittest.main()
