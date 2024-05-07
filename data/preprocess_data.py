import os
import subprocess

def clone_git_if_not_exist(repo_url='https://github.com/stacks/stacks-project', target_folder="stacks-project"):
    """
    Clone the git repository if the folder does not exist.
    """

    if not os.path.exists(target_folder):
        print(target_folder, " does not exist yet.")
        subprocess.run(['git', 'clone', repo_url, target_folder])
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

    try:
        with open(input_file, 'r') as in_file:
            with open(output_file, 'w') as out_file:
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


def read_chapters_files(input_file='chapter_files.txt', output_file='data.tex'):
    """
    Reads from a txt the names of latex files, clean the data, and concatenates their contents.
    """

    try:
        with open(input_file, 'r') as input_file:
            tex_files = [line.strip() for line in input_file if line.strip().endswith('.tex')]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

    with open(output_file, 'w') as out_file:

        for tex_file in tex_files:
            tex_file_path = os.path.join("stacks-project", tex_file)

            with open(tex_file_path, 'r') as input_tex:
                copy = False
                for input_line in input_tex:
                    # Filter empty lines
                    if input_line == "\n":
                        continue

                    # Filter end of chapter
                    if input_line[1:] == "input{chapters}\n":
                        break

                    if copy:
                        content = input_line.strip()
                        out_file.write(content + '\n')

                    # Filter begin of chapter
                    if input_line[1:] == "tableofcontents\n":
                        copy = True

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
    clone_git_if_not_exist()
    get_chapters_names(add_miscellany=False)
    read_chapters_files()