import unittest
import os
from utils import correct_latex_environments

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
