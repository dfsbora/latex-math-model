import unittest
import os
from utils import correct_latex_environments

class TestUtils(unittest.TestCase):

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

        result = correct_latex_environments(input_text)
        self.assertEqual(result.strip(), expected_output.strip())

    def test_unclosed_environment(self):
        input_text = r"""\begin{theorem}
given that X equal Y
"""

        expected_output = r"""\begin{theorem}
given that X equal Y
\end{theorem}
"""

        result = correct_latex_environments(input_text)
        self.assertEqual(result.strip(), expected_output.strip())

    def test_no_environment(self):
        input_text = r"""This is a regular text with no environments."""

        expected_output = input_text

        result = correct_latex_environments(input_text)
        self.assertEqual(result.strip(), expected_output.strip())


if __name__ == '__main__':
    unittest.main()
