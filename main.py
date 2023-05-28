import json
import os
import sys
from typing import Generator, List, Tuple

from colorama import Fore, init

from ast_util import ASTUtil
from tree_sitter import Language, Node, Parser, Tree

global file  # file being analyzed (relative path)
global file_path  # file being analyzed (full path)
global file_lines  # file lines (list of strings)

init(autoreset=True)  # colorama init


def traverse_tree_named(tree: Tree) -> Generator[Tuple[Node, int], None, None]:
    """
    Traverses a tree and yields each named node along with its depth.

    Parameters
    ----------
    tree : Tree
        The tree to traverse.

    Yields
    ------
    Tuple[Node, int]
        A tuple containing the node and its depth.
    """
    stack: list[tuple[Node, int]] = [(tree.root_node, 0)]
    while stack:
        node, depth = stack.pop()
        yield node, depth
        for child in reversed(node.named_children):
            stack.append((child, depth + 1))


def print_tree_named(tree: Tree):
    """
    Prints the named nodes of a tree in a hierarchical format.

    Parameters
    ----------
    tree : Tree
        The tree to print.
    """
    branches = set()
    for node, depth in traverse_tree_named(tree):
        prefix = ''.join(
            '│   ' if i in branches else '    ' for i in range(depth))
        if node.next_named_sibling is not None:
            branches.add(depth)
            prefix += '├── '
        elif node.parent is not None:
            if depth in branches:
                branches.remove(depth)
            prefix += '└── '

        print(prefix + node.type)


# def get_missing_nodes(tree: Tree) -> tuple[list[Node], list[Node]]:
#     missing_nodes: list[Node] = []
#     error_nodes: list[Node] = []

#     def traverse_tree(node: Node):
#         for n in node.children:
#             if n.is_missing:
#                 missing_nodes.append(n)
#             if n.has_error:
#                 print(f"error in {n.text}")
#                 print(n.sexp())
#                 error_nodes.append(n)

#             traverse_tree(n)

#     traverse_tree(tree.root_node)
#     return missing_nodes, error_nodes


def get_error_nodes(node: Node) -> Generator[Node, None, None]:
    """
    Traverses a tree starting from a given node and yields all nodes that have errors or are missing.
    The traversal takes into account both named or unnamed nodes.

    Parameters
    ----------
    node : Node
        The node to start the traversal from.
    Yields
    ------
    Node
        Nodes that have errors or are missing.
    """
    for n in node.children:
        if n.type == "ERROR" or n.is_missing:
            yield n
        if n.has_error:
            # there is an error inside this node let's check inside
            yield from get_error_nodes(n)


def print_error_line(er_line: int, padding: str, column_start: int, column_end: int, node_error: Node):
    """
    Prints the error line with a cursor pointing to the error location and the error message.

    Parameters
    ----------
    er_line : int
        The line number of the error.
    padding : str
        The padding to use before and after the error line.
    column_start : int
        The starting column of the error.
    column_end : int
        The ending column of the error.
    node_error : Node
        The node that contains the error.
    """
    print(f"{padding}{er_line}{padding}{file_lines[er_line-1]}")
    padding_with_line_number = " " * (len(f"{er_line}") + column_start-1)
    cursor_size = max(1, column_end - column_start)
    print(
        f"{padding * 2}{Fore.RED}{padding_with_line_number}{'~' * cursor_size}")

    if node_error.has_error and node_error.is_missing:
        error_message = f"{node_error.sexp()[1:-1]}"
    else:
        unexpected_tokens = "".join(n.text.decode('utf-8')
                                    for n in node_error.children)
        error_message = f"Unexpected token(s): {unexpected_tokens}"
    print(
        f"{padding * 2}{Fore.RED}{padding_with_line_number}{error_message}:")


def print_error(root_node: Node, error_type: str = "SYNTAX_ERROR") -> None:
    """
    Print error message for the given root node.

    Parameters
    ----------
    root_node : Node
        The root node of the syntax tree.
    error_type : str, optional
        The type of error to print, by default "SYNTAX_ERROR"
      """
    padding = " " * 5
    for node_error in get_error_nodes(root_node):
        er_line = node_error.start_point[0]+1
        column_start = node_error.start_point[1] + 1
        column_end = node_error.end_point[1] + 1
        print(
            f"{Fore.RED}{error_type}{Fore.RESET}:  {node_error.sexp()[1:-1]}")
        print(
            f"{padding}in file: '{file}:{er_line}:{column_start}:{column_end}', line: {er_line}", end=", ")
        print(
            f"from column {column_start} to {column_end}\n")
        print(f"{padding}{file_name}")
        if "--show-file" in sys.argv:
            print_file_with_errors(er_line, padding,
                                   column_start, column_end, node_error)
        else:
            print_error_line(er_line, padding, column_start,
                             column_end, node_error)


def print_file_with_errors(er_line: int, padding: str, column_start: int, column_end: int, node_with_error: Node) -> None:
    """
    Print the contents of a file with error highlighting.

    Parameters
    ----------
    er_line : int
        The line number where the error occurred.
    padding : str
        The padding to add before each line.
    column_start : int
        The starting column of the error.
    column_end : int
        The ending column of the error.
    node_with_error : Node
        The node containing the error message.
    """
    padding_factor = len(f"{len(file_lines)}")
    for i, line in enumerate(file_lines):
        i += 1
        if i == er_line:
            print_error_line(er_line, padding,
                             column_start, column_end, node_with_error)
        else:
            after_line_padding = "   " + " " * \
                (padding_factor - len(f"{i}"))

            print(f"{padding}{i}.{after_line_padding}{line}")


def traverse_statement(node: Node):
    match node.type:
        case 'let_statement':
            print(f"let statement {node.sexp()}")
            identifier = node.child_by_field_name('identifier')
            type_node = node.child_by_field_name('type')
            if (identifier == None) or (type_node == None):
                print("missing if condition or body")
                exit(-1)
            print(
                f"Analyzing let statement with identifier: {identifier.text} and type: {type_node.text}")
            # Analyze let statement
        case 'if_statement':
            condition = node.child_by_field_name('if_condition')
            body = node.child_by_field_name("if_body")
            if (condition == None) or (body == None):
                print("missing if condition or body")
                exit(-1)
            print(
                f"Analyzing if statement with condition: {condition.type} and body: {body.type}")
            traverse_expression(condition)
            traverse_statement(body)
            # Analyze if statement
        # ... handle other statement types ...
        case 'do_while_statement':
            print(f" do while {node.sexp()}")
        case 'return_statement':
            return_val = node.child_by_field_name('return_value')
            if return_val == None:
                print("return with value")
            else:
                traverse_expression(return_val)
                print(f"return with no value")
        case 'assignment_statement':
            print(f"assignment {node.sexp()}")
        case 'input_statement':
            print(f"input {node.sexp()}")
        case 'print_statement':
            print(f"print {node.sexp()}")
        case 'function_call':
            print(f"function call statement {node.sexp()}")
        case 'post_increment_statement':
            print(f"postincrem {node.sexp()}")
        case "ERROR":
            print_error(node)
        case _:
            print(f"unrecognized statement{node.type}")
            for child in node.children:
                traverse_statement(child)


def traverse_expression(node: Node):

    match node.type:
        case 'or_expression':
            children = node.children_by_field_name('_expression')
            left = children[0]
            right = children[1]
            print(
                f"Analyzing or expression with left: {left} and right{right}")
            # Analyze or expression
        case 'equality_expression':
            left = node.child_by_field_name('left')
            right = node.child_by_field_name('right')
            print(
                f"Analyzing equality expression with left: {left} and right{right}")
        # ... handle other expression types ...
        case 'addition_expression':
            children = node.children_by_field_name('_expression')
            left = children[0]
            right = children[1]
            print(
                f"Analyzing addition expression with left: {left} and right{right}")
            # Analyze addition expression

        case 'literal_number':
            print(f"number {node.sexp()}")
        case 'literal_string':
            print(f"string {node.sexp()}")
        case 'identifier':
            print(f"esto es un id {node.text}")
        case 'literal_boolean':
            print(f"boolean {node.sexp()}")

        case "ERROR":
            print(Fore.RED + f"ERROR {node.sexp()}")
        case _:
            print(f"unrecognized expression {node.type}")
            for child in node.children:
                traverse_expression(child)


def traverse(node: Node):
    print("Analyzing program")
    for child in node.named_children:
        traverse_statement(child)


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
    global file_name
    global file_lines
    global file
    if len(sys.argv) < 2:
        print("Usage: python main.py <file>")
        exit(-1)
    # get argv[1] and check for existance
    file = sys.argv[1]
    if file == "--help" or file == "-h" or file == "-help":
        print_usage()
        exit(-1)
    if not os.path.exists(file):
        print(f"File {file} does not exist")
        exit(-1)

    if os.path.isdir(file):
        print(f"File {file} is a directory")
        exit(-1)
    with open(file, 'r') as f:
        file_name = os.path.abspath(file)
        file_lines_raw = f.readlines()
        file_lines = [line.replace("\n", "") for line in file_lines_raw]
        return ''.join(file_lines_raw)


def build_language_and_create_parser() -> Parser:
    """
    Build the JSPDL language library using Tree-sitter and create the parser with 
    the language set.

    Returns
    -------
    Parser
        The parser object.

    """
    Language.build_library(
        # Store the library in the `build` directory
        'jspdl_build/jspdl.so',

        # Include one or more languages
        [
            'jspdl_language',
        ]
    )

    JSPDL_LANGUAGE = Language('jspdl_build/jspdl.so', 'jspdl')
    parser = Parser()
    parser.set_language(JSPDL_LANGUAGE)
    return parser


def type_check(tree: Tree):
    if tree.root_node.has_error:
        print_error(tree.root_node)
        exit(-1)

    traverse(tree.root_node)


def json_tree_print(tree: Tree, code: str) -> None:
    """
    Print the simplified AST of the given tree in JSON format.

    Parameters
    ----------
    tree : Tree
        The tree to simplify and print.
    code : str
        The code corresponding to the tree.
    """
    ast_util = ASTUtil()
    simplified_ast = ast_util.simplify_ast(tree, code)
    s_ast_json = json.dumps(simplified_ast, indent=4)
    print(s_ast_json)


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
    parser = build_language_and_create_parser()
    code = parse_arguments_and_get_raw_code()
    tree = parser.parse(bytes(code, 'utf8'))
    execute_options(tree, code)
    type_check(tree)


if __name__ == "__main__":
    main()
