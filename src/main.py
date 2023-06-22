
import os
import sys

from colorama import Fore, init
from tree_sitter import Tree

import globals
from ast_util import json_tree_print, print_tree_named
from code_gen import gen_code
from errors import check_parsing_errors
from language import parser
from type_checker import ast_type_check_tree

init(autoreset=True)  # colorama init


def print_usage():
    print(
        """
Usage: python main.py <file> [options]
options:
    --ast: show ast
    --show-file: show file with errors when errors are found
""")


def parse_arguments_and_get_raw_code() -> str:
    """
    Parse command line arguments and return the contents of the file as a string.

    Returns
    -------
    str
        The contents of the file as a string.

    Raises
    ------
    SystemExit
        If the command line arguments are invalid or the file does not exist.
    """

    if len(sys.argv) < 2:
        print("Usage: python main.py <file>")
        exit(-1)
    # get argv[1] and check for existance
    globals.file = sys.argv[1]
    if globals.file == "--help" or globals.file == "-h" or globals.file == "-help":
        print_usage()
        exit(-1)
    if not os.path.exists(globals.file):
        print(f"File {globals.file} does not exist")
        exit(-1)

    if os.path.isdir(globals.file):
        print(f"File {globals.file} is a directory")
        exit(-1)
    if "-o" in sys.argv:
        try:
            output_file = sys.argv[sys.argv.index("-o") + 1]
            if ".ens" not in output_file:
                print("Output file must be a .ens file")
                exit(-1)
            globals.output_file = output_file
        except IndexError:
            print("No output file specified")
            exit(-1)

    with open(globals.file, 'r') as f:

        globals.file_path = os.path.abspath(globals.file)
        globals.file_name = os.path.basename(globals.file_path).__str__()
        file_lines_raw = f.readlines()
        globals.file_lines = [line.replace("\n", "")
                              for line in file_lines_raw]
        return ''.join(file_lines_raw)


def execute_options(tree: Tree, code: str) -> None:
    """
    Execute the command line options specified by the user.

    Parameters
    ----------
    tree : Tree
        The syntax tree of the code.
    code : str
        The code to execute.
    """
    if ("--json-ast" in sys.argv):
        json_tree_print(tree, code)
    if ("--ast" in sys.argv):
        print("AST:")
        print_tree_named(tree)


def main() -> None:
    """
    Parse command line arguments, build the language library, parse the code, execute command line options, and perform type checking.
    """
    code = parse_arguments_and_get_raw_code()
    tree = parser.parse(bytes(code, 'utf8'))
    execute_options(tree, code)
    check_parsing_errors(tree)
    if not ast_type_check_tree(tree):
        print(f"{Fore.RED}Type checking failed")
        print("Code generation aborted, errors need to be fixed in order to generate code")
        exit(-1)
    gen_code()


if __name__ == "__main__":
    main()
