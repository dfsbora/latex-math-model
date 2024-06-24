import re
from collections import Counter
import argparse


def mask_formulas(text, masking_word):
    """
    Masks latex formulas in the given text with a specified masking word.

    Parameters:
    - text (str): The input text containing latex formulas to mask.
    - masking_word (str): The word to replace latex formulas with.

    Returns:
    - str: The text with latex formulas replaced by the masking word.
    """

    pattern_single = r'(?<!\\)\$[^$]*\$(?!\$)'
    pattern_double = r'(?<!\\)\$\$[^$]*\$\$(?!\$)'

    new_single = '$' + masking_word + '$'
    new_double = '$$' + masking_word + '$$'

    single_matches = re.findall(pattern_single, text)
    for match in single_matches:
        replacement = new_single if len(match) > 2 else match
        text = text.replace(match, replacement)

    double_matches = re.findall(pattern_double, text)
    for match in double_matches:
        replacement = new_double if len(match) > 4 else match
        text = text.replace(match, replacement)

    return text


def process_line(line,masking_word):
    matches = []
    for command in latex_commands:
        matches.extend(re.finditer(command, line))
    matches.extend(re.finditer(formula_pattern, line))

    # Sort matches by start position
    matches.sort(key=lambda x: x.start())

    # Replace non-matching words with "TEXT"
    result = []
    last_end = 0
    for match in matches:
        start, end = match.span()
        if last_end < start:
            # Process the text between matches
            non_matching_text = line[last_end:start]
            # Split the text by spaces
            words = non_matching_text.split()
            # Replace words with "TEXT"
            result.extend([masking_word] * len(words))
        # Append the matched LaTeX command or formula
        result.append(match.group())
        last_end = end

    # Process any remaining text after the last match
    if last_end < len(line):
        non_matching_text = line[last_end:]
        words = non_matching_text.split()
        result.extend([masking_word] * len(words))

    # Remove consecutive "TEXT" entries
    final_result = []
    prev_was_text = False
    for part in result:
        if part == masking_word:
            if not prev_was_text:
                final_result.append(part)
            prev_was_text = True
        else:
            final_result.append(part)
            prev_was_text = False

    # Join the final result into a string
    return ' '.join(final_result)


latex_commands = [
    r'\\cite\{[^}]*\}', r'\\begin\{[^}]*\}', r'\\end\{[^}]*\}', r'\\section\{[^}]*\}',
    r'\\subsection\{[^}]*\}', r'\\textbf\{[^}]*\}', r'\\textit\{[^}]*\}', r'\\underline\{[^}]*\}',
    r'\\item', r'\\includegraphics\{[^}]*\}', r'\\ref\{[^}]*\}', r'\\label\{[^}]*\}'
]

# Regular expression for formulas enclosed in $
formula_pattern = r'\$[^$]*?\$'

def mask_text(text, masking_word):

    output_lines = []
    flag_formula = False

    for line in text.splitlines():
        stripped_line = line.strip()

        #Begin and end of formula
        if stripped_line == "$$":
            output_lines.append(line)
            flag_formula = not flag_formula

        #Copy whole formula
        elif flag_formula:
            output_lines.append(line)

        else:

            processed_line = process_line(line,masking_word)
            output_lines.append(processed_line)
    return '\n'.join(output_lines)




def mask_textold(text, masking_word):
    """
    Masks non-latex lines or non-latex parts of a line

    Parameters:
    - text (str): The input text
    - masking_word (str): The word to replace non-latex lines with.

    Returns:
    - str: The text with non-latex parts replaced by the masking word and latex formulas preserved.
    """

    latex_command_pattern = r'\\[a-zA-Z]+(\{[^{}]*\})*'
    output_lines = []
    flag_formula = False
    flag_item = False

    for line in text.splitlines():
        stripped_line = line.strip()

        #Begin and end of formula
        if stripped_line == "$$":
            output_lines.append(line)
            flag_formula = not flag_formula

        #Copy whole formula
        elif flag_formula:
            output_lines.append(line)

        #Remaining kind of lines
        else:
            masked_parts = []
            parts = stripped_line.split('$')

            for i, part in enumerate(parts):
                #Deal with part of the line that is not the inline formula
                if i % 2 == 0:
                    processed_part = []
                    if part.startswith("\\item"):
                        processed_part.append("\\item " + masking_word)
                    else:
                        words = part.split('\\')
                        processed_part = []
                        for j, word in enumerate(words):
                            if j % 2 == 0 and word:
                                #Mask this part because there is no command in it anyway
                                processed_part.append(masking_word)

                            else:
                                subwords = word.split('}')
                                processed_word = []
                                for k, subword in enumerate(subwords):

                                    if subword:
                                        # Preserve the latex commands
                                        if k % 2 == 0:
                                            processed_word.append("\\" + subword + "}")
                                        #Mask the rest
                                        else:
                                            processed_word.append(masking_word)

                                processed_word = ' '.join(processed_word)
                                processed_part.append(processed_word)
                    processed_part = ' '.join(processed_part)
                    masked_parts.append(processed_part)

                #Copy inline formula
                else:
                    masked_parts.append("$"+part+"$")
            masked_line = ' '.join(masked_parts)
            output_lines.append(masked_line)

    return '\n'.join(output_lines)

def file_to_text(input_file):
    with open(input_file, 'r', encoding='utf-8') as in_file:
        text = in_file.read()
    return text

def text_to_file(text, output_file):
    with open(output_file, 'w') as out_file:
        out_file.write(text)

def run_mask_formulas(input_filename, output_filename, masking_word):
    """
    Reads latex text from an input file, masks latex formulas using a specified masking word,
    and writes the masked text to an output file.
    """
    text = file_to_text(input_filename)
    masked_text = mask_formulas(text, masking_word)
    text_to_file(masked_text, output_filename)

def run_mask_text(input_filename, output_filename, masking_word):
    """
    Reads text from an input file, masks non-LaTeX lines and alternates between
    masking and preserving LaTeX formulas, and writes the masked text to an output file.
    """
    text = file_to_text(input_filename)
    masked_text = mask_text(text, masking_word)
    text_to_file(masked_text, output_filename)


################################
def find_math_commands(tex_content):
    """
    Find all math commands in LaTeX content.

    Args:
    - tex_content (str): The latex content to search for commands.

    Returns:
    - Counter: A Counter object containing the frequency of each LaTeX command found.
    """

    math_notation_pattern = re.compile(r'\$.*?\$|\$\$.*?\$\$', re.DOTALL)
    command_pattern = re.compile(r'\\[a-zA-Z]+')

    math_notations = math_notation_pattern.findall(tex_content)

    commands = []
    for notation in math_notations:
        commands.extend(command_pattern.findall(notation))

    command_frequency = Counter(commands)

    sorted_commands = sorted(command_frequency.items(), key=lambda item: item[1], reverse=True)

    return sorted_commands

def run_math_command_frequencies(input_file):
    """
     Read latex file and return math commands frequencies.
    """
    content = file_to_text(input_file)
    command_frequency = find_math_commands(content)
    return command_frequency


def print_command_frequency(command_frequency):
    for command, frequency in command_frequency:
        print(f"{command} {frequency}")

def save_command_frequency(command_frequency, filename):
    with open(filename, 'w') as file:
        for command, frequency in command_frequency:
            file.write(f"{command} {frequency}\n")

def merge_commands(content):
    """
     Merge similar latex math commands in the content.

     The merge_mapping was generated by chatgpt from the math commands frequencies
     """
    merge_mapping = {
        # Merging font styles
        '\\mathcal': '\\mathcal',
        '\\mathfrak': '\\mathcal',
        '\\mathbf': '\\mathbf',
        '\\mathit': '\\mathbf',
        '\\mathrm': '\\mathrm',

        # Merging logical and set theoretical symbols
        '\\subset': '\\subset',
        '\\subseteq': '\\subset',
        '\\supset': '\\supset',
        '\\supseteq': '\\supset',
        '\\notin': '\\notin',
        '\\not\\in': '\\notin',
        '\\neg': '\\neg',
        '\\lnot': '\\neg',
        '\\forall': '\\forall',
        '\\exists': '\\exists',

        # Merging algebra symbols
        '\\times': '\\times',
        '\\cdot': '\\times',
        '\\circ': '\\circ',
        '\\bullet': '\\circ',
        '\\oplus': '\\oplus',
        '\\star': '\\oplus',
        '\\otimes': '\\otimes',
        '\\diamond': '\\otimes',

        # Merging miscellaneous symbols
        '\\to': '\\to',
        '\\gets': '\\to',
        '\\rightarrow': '\\to',
        '\\leftarrow': '\\to',
        '\\leadsto': '\\leadsto',
        '\\longrightarrow': '\\leadsto',
        '\\leftrightarrow': '\\leftrightarrow',
        '\\longleftrightarrow': '\\leftrightarrow',
        '\\iff': '\\leftrightarrow',
        '\\Leftrightarrow': '\\leftrightarrow',
        '\\mod': '\\bmod',
        '\\bmod': '\\bmod'
    }
    for old_cmd, new_cmd in merge_mapping.items():
        content = content.replace(old_cmd, new_cmd)
    return content


def run_merge_commands(input_filename, output_filename):
    """
     Reads file and merges similar latex math commands.
    """

    content = file_to_text(input_filename)
    content = merge_commands(content)
    text_to_file(content, output_filename)


def run_functions_on_file(input_file, output_file, mask_text_flag, mask_formulas_flag, merge_math_commands_flag, chunk_size):
    """
    Read a file in chunks, process each chunk, and write to an output file.

    Parameters:
    - input_file (str): Path to the input file.
    - output_file (str): Path to the output file.
    - chunk_size (int): The size of each chunk to be read.
    - args
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        while True:
            text = infile.read(chunk_size)
            if not text:
                break
            if mask_text_flag:
                print("Masking text...")
                text = mask_text(text, "TEXT")
                print("Text masking complete")

            if mask_formulas_flag:
                print("Masking formulas...")
                text = mask_formulas(text, "FORMULA")
                print("Formulas masking complete")

            elif merge_math_commands_flag:
                print("Merging commands...")
                text = merge_commands(text)
                print("Commands merge complete")

            outfile.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplify the data")
    parser.add_argument('--input_file', type=bool, default='data.tex', help='Original dataset file name.')
    parser.add_argument('--output_file', type=str, default='data_simplified.tex', help='Final data file name.')
    parser.add_argument('--mask_formulas', action='store_true', help='Mask formulas')
    parser.add_argument('--mask_text', action='store_true', help='Mask text')
    parser.add_argument('--merge_math_commands', action='store_true', help='Merge math commands in the formulas')
    args = parser.parse_args()

    if not args.mask_formulas and not args.mask_text and not args.merge_math_commands:
        print("No data simplification method selected.\nFinishing execution.")
        exit()

    run_functions_on_file(args.input_file, args.output_file, args.mask_text, args.mask_formulas, args.merge_math_commands, chunk_size=1024)
