import os
import subprocess
import argparse

def clone_git_if_not_exist(repo_url='https://github.com/stacks/stacks-project', target_folder="stacks-project"):
    """
    Clone the git repository if the folder does not exist.
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_folder_abs = os.path.join(current_dir, target_folder)

    if not os.path.exists(target_folder_abs):
        print(target_folder, " does not exist yet.")
        subprocess.run(['git', 'clone', repo_url, target_folder_abs])
        print("Cloning successful")
    else:
        print(target_folder, " already exists.")

def get_chapters_names(input_file='stacks-project/chapters.tex', output_file='chapter_files.txt', add_miscellany=False):
    """
    Extracts the names of chapters from the book's table of contents and writes them to a file.

    Parameters:
    add_miscellany (bool): Whether to include chapters after the "Miscellany" section.
                            (Not algebra content). Default is False.
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_abs = os.path.join(current_dir, input_file)
    output_file_abs = os.path.join(current_dir, output_file)

    try:
        with open(input_file_abs, 'r') as in_file:
            with open(output_file_abs, 'w') as out_file:
                for line in in_file:
                    if line.strip()[:16] == "\item \hyperref[":
                        chapter_name = line.split('[')[1].split(']')[0]
                        file_name = chapter_name[:-16] + '.tex'
                        out_file.write(file_name + '\n')

                    #Filter "Miscellany" section (not algebra content)
                    elif not add_miscellany and line.strip()=="Miscellany":
                        break
        print("Filtered chapter files: ", output_file)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise


def read_chapters_files(input_file='chapter_files.txt', output_file='data.tex', include_eos=True):
    """
    Reads from a txt the names of latex files, clean the data, and concatenates their contents.
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_abs = os.path.join(current_dir, input_file)
    output_file_abs = os.path.join(current_dir, output_file)


    try:
        with open(input_file_abs, 'r') as input_file:
            tex_files = [line.strip() for line in input_file if line.strip().endswith('.tex')]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise


    eos_flag = False
    with open(output_file_abs, 'w') as out_file:

        for tex_file in tex_files:
            tex_file_path = os.path.join("stacks-project", tex_file)

            try:
                with open(tex_file_path, 'r') as input_tex:
                    copy = False
                    for input_line in input_tex:
                        # Filter end of chapter
                        if input_line[1:] == "input{chapters}\n":
                            break

                        if copy:
                            # Filter empty lines
                            if input_line == "\n":
                                # Include tag of eos
                                if eos_flag and include_eos:
                                    out_file.write("eos ")
                                    eos_flag = False
                                else:
                                    pass

                            # Copy content
                            else:
                                content = input_line.strip()
                                out_file.write(content + '\n')
                                #todo decide if really keeps no eos after label
                                if content[:7] == "\label{":
                                    eos_flag = False
                                else:
                                    eos_flag = True

                        # Filter begin of chapter
                        if input_line[1:] == "tableofcontents\n":
                            copy = True
                            eos_flag = False
            except:
                pass

    print("Concatenated clean data:  ", output_file)



def latex_to_text(input_file, output_file):
    """
    Convert a latex file to plain text using pandoc.
    """

    try:
        # Run pandoc command
        subprocess.run(['pandoc', '-f', 'latex', '-t', 'plain', input_file, '-o', output_file], check=True)
        print("Conversion to plain text successful")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the data")
    parser.add_argument('--add_miscellany', type=bool, default=False, help='Whether to include chapters after the "Miscellany" section.')
    parser.add_argument('--add_eos', type=bool, default=False, help='Whether to include EOS tokens.')
    parser.add_argument('--output_file', type=str, default='data.tex', help='Final data file name.')
    args = parser.parse_args()

    clone_git_if_not_exist()
    get_chapters_names(add_miscellany=args.add_miscellany)
    read_chapters_files(output_file=args.output_file, include_eos=args.add_eos)
