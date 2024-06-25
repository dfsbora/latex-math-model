import re


def generate_config_dict(arguments, architecture_name='LSTM'):
    """
    Generate config dictionary for wandb
    :param arguments:
    :param architecture_name:
    :return:
    """

    dict = vars(arguments)
    keys = list(dict.keys())
    keys = keys[:-3]
    dict = {key: dict[key] for key in keys}
    dict['architecture'] = architecture_name
    return dict


def correct_latex_environments(text):
    begin_pattern = r'\\begin\{([^\}]+)\}'
    end_pattern = r'\\end\{([^\}]+)\}'

    stack = []
    output_lines = []

    num_corrected_env = 0
    num_closed_env = 0
    num_deleted_end = 0
    flag_deleted = False

    for line in text.splitlines():
        # Find all begin and end matches in the current line
        begin_matches = list(re.finditer(begin_pattern, line))
        end_matches = list(re.finditer(end_pattern, line))

        for match in begin_matches:
            environment = match.group(1)
            stack.append(environment)

        for match in end_matches:
            environment = match.group(1)
            if stack and stack[-1] == environment:
                stack.pop()

            elif stack:
                correct_environment = stack.pop()
                corrected_line = f'\\end{{{correct_environment}}}'
                line = line.replace(match.group(0), corrected_line, 1)
                num_corrected_env += 1

            else:
                num_deleted_end += 1
                flag_deleted = True
                continue

        if flag_deleted:
            flag_deleted = False
            continue

        output_lines.append(line)

    # Closes unclosed environments
    for environment in reversed(stack):
        output_lines.append(f'\\end{{{environment}}}')
        num_closed_env += 1

    return '\n'.join(output_lines), num_corrected_env, num_closed_env, num_deleted_end


class EvaluatePrompt:
    def __init__(self, text, cites_file="data/cites.txt", labels_file="data/labels.txt"):
        """
        Args:
           text (str): The text to evaluate.
           cites_file (str): File path to the citations list.
           labels_file (str): File path to the labels list.
        """
        self.text = text
        self.cites_list = self._file_to_content(cites_file)
        self.labels_list = self._file_to_content(labels_file)

        # Regular expressions for parsing LaTeX formulas, citations, labels, and references
        self.pattern_single = re.compile(r'(?<!\\)\$(?!\$)(.*?)(?<!\\)\$(?!\$)')
        self.pattern_double = re.compile(r'(?<!\\)\$\$(.*?)(?<!\\)\$\$', re.DOTALL)
        self.cite_pattern = re.compile(r'\\cite\{(.*?)\}')
        self.label_pattern = re.compile(r'\\label\{(.*?)\}')
        self.ref_pattern = re.compile(r'\\ref\{(.*?)\}')

        # Dictionary mapping delimiters
        self.open_close_pairs = {
            '(': ')',
            '{': '}',
            '[': ']',
            '\\left(': '\\right)',
            '\\left{': '\\right}',
            '\\left[': '\\right]'
        }

    def set_text(self, new_text):
        self.text = new_text

    @staticmethod
    def _file_to_content(input_file):
        with open(input_file, 'r', encoding='utf-8') as file:
            content = set(line.strip() for line in file)
        return content

    def _parse_formulas(self):
        """
        Returns:
            list: List of LaTeX formulas found in the text.
        """
        double_matches = self.pattern_double.findall(self.text)
        single_matches = self.pattern_single.findall(self.text)
        all_formulas = single_matches + double_matches
        return all_formulas

    def _parse_refs(self):
        matches = self.ref_pattern.findall(self.text)
        return matches

    def _parse_cites(self):
        matches = self.cite_pattern.findall(self.text)
        return matches

    def _parse_labels(self):
        matches = self.label_pattern.findall(self.text)
        return matches

    def compare_labels_with_original_labels(self):
        """
        Compare labels found in the prompt text with original labels.

        Returns:
            tuple: Number of generated labels, common labels, and original labels not found in generated.
        """
        prompt_labels = self._parse_labels()
        common = set(prompt_labels) & self.labels_list

        num_generated = len(set(prompt_labels))
        num_copy = len(common)
        num_original = num_generated - num_copy
        return num_generated, num_copy, num_original

    def compare_cites_with_original_cites(self):
        """
        Compare citations found in the prompt text with original citations.

        Returns:
            tuple: Number of generated citations, common citations, and original citations not found in generated.
        """
        prompt_cites = self._parse_cites()
        common = set(prompt_cites) & self.cites_list
        num_generated = len(set(prompt_cites))
        num_copy = len(common)
        num_original = num_generated - num_copy
        return num_generated, num_copy, num_original

    def check_refs_with_prompt_labels(self):
        """
        Check the references found in the prompt text against prompt labels.

        Returns:
            tuple: Number of generated references, common references, and references not corresponding to prompt labels.
        """
        prompt_refs = self._parse_refs()
        prompt_labels = self._parse_labels()
        common = set(prompt_labels) & set(prompt_refs)

        num_generated = len(set(prompt_refs))
        num_copy = len(common)
        num_original = num_generated - num_copy
        return num_generated, num_copy, num_original

    def check_refs_with_original_labels(self):
        """
        Check the references found in the prompt text against original labels.

        Returns:
            tuple: Number of generated references, common references,
            and references not corresponding to original labels.
        """
        prompt_refs = self._parse_refs()
        common = self.labels_list & set(prompt_refs)

        num_generated = len(set(prompt_refs))
        num_copy = len(common)
        num_original = num_generated - num_copy
        return num_generated, num_copy, num_original

    def check_valid_formulas(self):
        """
        Check validity of LaTeX formulas found in the prompt text.
        At the moment only evaluates the balance of delimiters.

        Returns:
            tuple: Number of generated formulas and number of valid formulas.
        """
        prompt_formulas = self._parse_formulas()
        num_generated = len(prompt_formulas)
        num_valid = 0
        for formula in prompt_formulas:
            if self._is_balanced(formula):
                num_valid += 1
        return num_generated, num_valid

    def _is_balanced(self, latex_formula):
        """
        Check if a LaTeX formula is balanced with respect to delimiters.

        Returns:
            bool: True if balanced, False otherwise.
        """
        stack = []
        i = 0
        while i < len(latex_formula):
            char = latex_formula[i]

            if latex_formula[i:i + 6] == '\\left(' or latex_formula[i:i + 6] == '\\left{' or latex_formula[
                                                                                             i:i + 6] == '\\left[':
                stack.append(latex_formula[i:i + 6])
                i += 6

            elif latex_formula[i:i + 7] == '\\right)' or latex_formula[i:i + 7] == '\\right}' or latex_formula[
                                                                                                 i:i + 7] == '\\right]':
                if not stack:
                    return False
                top = stack.pop()
                if self.open_close_pairs[top] != latex_formula[i:i + 7]:
                    return False
                i += 7
            elif char in self.open_close_pairs:
                stack.append(char)
                i += 1
            elif char in self.open_close_pairs.values():
                if not stack:
                    return False
                top = stack.pop()
                if self.open_close_pairs[top] != char:
                    return False
                i += 1
            else:
                i += 1
        return not stack


if __name__ == "__main__":
    input_text = r"""
    This is a sample text with some formulas.
    Here is one: $a^2 + b^2 = c^2$ \cite{dag12} and
    here is another: $$E=mc^2$$ \cite{cite4}..
    """
    evaluator = EvaluatePrompt(input_text)
    num_generated, num_copy, num_original = evaluator.compare_cites_with_original_cites()
    print(num_generated, num_copy, num_original)

    new_text = r"""
    This is a label \label{label1} and here is
    another: \label{remark-morphism-topoi-big}.
    """
    evaluator.set_text(new_text)
    num_generated, num_copy, num_original = evaluator.compare_labels_with_original_labels()
    print(num_generated, num_copy, num_original)
