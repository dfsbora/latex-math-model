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

