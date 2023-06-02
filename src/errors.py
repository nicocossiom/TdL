import sys
from enum import Enum
from typing import Generator

from colorama import Fore
from tree_sitter import Node, Tree

from ast_util import unwrap_text
from symbol_table import JSPDLType


class Error:
    def __init__(self, node: Node):
        self.node: Node = node

    def __repr__(self):
        return f"{Fore.RED}{self.__class__.__name__}{Fore.RESET}"


class SyntaxError(Error):
    def __init__(self, node, message=None):
        super().__init__(node)

    def __repr__(self):
        return f"{super().__repr__()}: {self.node.sexp()[1:-1]}"


class UndeclaredVariableError(Error):
    def __init__(self, node: Node):
        super().__init__(node)
        self.identifier = self.node.child_by_field_name("identifier")

    def __repr__(self):
        assert self.identifier is not None
        return f"{super().__repr__()}: variable '{unwrap_text(self.identifier.text)}' is not declared in any scope"


class PreDeclarationError(Error):
    def __init__(self, node: Node, predeclared_node: Node):
        super().__init__(node)

    def __repr__(self):
        return f"{super().__repr__()}: {unwrap_text(self.node.text)} is already declared in the current scope"


class TypeMismatchError(Error):

    def __init__(self, node: Node, expected_type: JSPDLType, actual_type: JSPDLType):
        super().__init__(node)
        self.expected_type = expected_type
        self.actual_type = actual_type

    def __repr__(self):
        return f"{super().__repr__()}: expected type {self.expected_type} but got {self.actual_type}"


class NonInitializedError(Error):
    def __init__(self, node: Node):
        super().__init__(node)
        identifier_text = self.node.text
        self.identifier = unwrap_text(identifier_text)

    def __repr__(self):
        return super().__repr__() + f": variable '{self.identifier}' has not yet been initialized"


def check_parsing_errors(tree: Tree):
    if tree.root_node.has_error:
        print_parsing_errors(tree.root_node)
        exit(-1)


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


def print_error(error: Error):
    """
    Prints an error message.

    Parameters
    ----------
    error : Error
        The error to print.
    """
    import globals
    file_path = globals.file_path
    file_name = globals.file_name

    indentation = " " * 5
    line_number = error.node.start_point[0]+1
    start_column = error.node.start_point[1] + 1
    end_column = error.node.end_point[1] + 1
    print(f"\n{Fore.RED}{error}{Fore.RESET}")
    print(f"{indentation}in file: '{file_path}:{line_number}:{start_column}:{end_column}', line: {line_number}", end=", ")
    print(f"from column {start_column} to {end_column}\n")
    print(f"{indentation}{file_name}")
    print_error_line(line_number, indentation,
                     start_column, end_column, error.node)


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
    import globals
    file_lines = globals.file_lines
    assert file_lines is not None
    max_line_length = max(len(line) for line in file_lines)
    print(f"{padding}{'-'*max_line_length}")
    print(f"{padding}{er_line}{padding}{file_lines[er_line-1]}")
    padding_with_line_number = " " * (len(f"{er_line}") + column_start-1)
    cursor_size = max(1, column_end - column_start)
    print(
        f"{padding * 2}{Fore.RED}{padding_with_line_number}{'~' * cursor_size}")

    error_message = ""
    if node_error.has_error and node_error.is_missing:
        error_message = f"{node_error.sexp()[1:-1]}"
    elif node_error.is_missing:
        unexpected_tokens = "".join(n.text.decode('utf-8')
                                    for n in node_error.children)
        error_message = f"Unexpected token(s): {unexpected_tokens}"
    print(
        f"{padding * 2}{Fore.RED}{padding_with_line_number}{error_message}")


def print_parsing_errors(node: Node) -> None:
    """
    Print error message for the given root node.

    Parameters
    ----------
    root_node : Node
        The root node of the syntax tree.
    error : str, optional
        The type of error to print, by default "SYNTAX_ERROR"
      """
    import globals
    file_name = globals.file_name
    file_path = globals.file_path
    indentation = " " * 5
    for erroneous_node in get_error_nodes(node):
        line_number = erroneous_node.start_point[0]+1
        start_column = erroneous_node.start_point[1] + 1
        end_column = erroneous_node.end_point[1] + 1
        error = SyntaxError(erroneous_node)
        print(
            f"\n{Fore.RED}{error}{Fore.RESET}")
        print(
            f"{indentation}in file: '{file_path}:{line_number}:{start_column}:{end_column}', line: {line_number}", end=", ")
        print(
            f"from column {start_column} to {end_column}\n")
        print(f"{indentation}{file_name}")
        if "--show-file" in sys.argv:
            print_file_with_errors(line_number, indentation,
                                   start_column, end_column, erroneous_node)
        else:
            print_error_line(line_number, indentation, start_column,
                             end_column, erroneous_node)


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
    import globals
    file_lines = globals.file_lines
    assert file_lines is not None
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
