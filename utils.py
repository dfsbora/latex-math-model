import re
from collections import Counter
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
    def __init__(self, text, cites_file = "data/cites.txt", labels_file = "data/labels.txt"):
        self.text = text
        #TODO add comparison of formulas
        #TODO add comparison of refs to labels in prompt and in original dataset
        self.cites_list = self._file_to_content(cites_file)
        self.labels_list = self._file_to_content(labels_file)

        #self.pattern_single = re.compile(r'(?<!\\)\$(?!\$)(.*?)(?<!\\)\$(?!\$)')
        #self.pattern_double = re.compile(r'(?<!\\)\$\$(.*?)(?<!\\)\$\$', re.DOTALL)
        self.cite_pattern = re.compile(r'\\cite\{(.*?)\}')
        self.label_pattern = re.compile(r'\\label\{(.*?)\}')

    def set_text(self, new_text):
        self.text = new_text

    def _file_to_content(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as file:
            content = set(line.strip() for line in file)
        return content

    def _parse_formulas(self):
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
        # Find labels in the outputted text
        prompt_labels = self._parse_labels()
        # Compare labels with list of original dataset
        common = set(prompt_labels) & self.labels_list

        num_generated = len(set(prompt_labels))
        num_copy = len(common)
        num_original = num_generated - num_copy
        return num_generated, num_copy, num_original


    def compare_cites_with_original_cites(self):
        # Find labels in the outputted text
        prompt_cites = self._parse_cites()
        # Compare labels with list of original dataset
        common = set(prompt_cites) & self.cites_list

        num_generated = len(set(prompt_cites))
        num_copy = len(common)
        num_original = num_generated - num_copy
        return num_generated, num_copy, num_original





if __name__ == "__main__":
    input_text = """
    This is a sample text with some formulas. Here is one: $a^2 + b^2 = c^2$ \cite{dag12} and here is another: $$E=mc^2$$ \cite{cite4}.
    Another inline formula is $k^* = Lk^*$.
    """

    # Instantiate the class with the input text
    evaluator = EvaluatePrompt(input_text)

    # Compare parsed formulas with another text file
    num_generated, num_copy, num_original = evaluator.compare_cites_with_original_cites()
    print(num_generated)
    print(num_copy)
    print(num_original)
