from typing import Callable

from tree_sitter import Node

from ast_util import unwrap_text
from errors import PreDeclarationError, UndeclaredError, print_error
from language import language
from symbol_table import FnEntry, VarEntry, current_symbol_table, symbol_table


def let_statement(node: Node):
    query = language.query(
        "(let_statement type: (type) @type identifier: (identifier) @identifier)")
    captures = query.captures(node)
    let_type = unwrap_text(captures[0][0].text)
    let_identifier = unwrap_text(captures[1][0].text)
    if let_identifier not in symbol_table:
        symbol_table[let_identifier] = VarEntry(
            type=let_type, value=None, offset=None, node=node)
    else:
        pred_node = symbol_table[let_identifier]
        assert isinstance(pred_node, VarEntry) and pred_node.node != None
        print_error(PreDeclarationError(node, pred_node.node))


def assignment_statement(node: Node):
    query = language.query(
        "(assignment_statement (identifier) @identifier (expression) @expression)")
    captures = query.captures(node)
    identifier = unwrap_text(captures[0][0].text)
    expression = captures[1][0]
    if identifier not in current_symbol_table or identifier not in symbol_table:
        print_error(UndeclaredError(node))
    else:
        print(f"assignment statement {expression.sexp()}")


rule_functions: dict[str, Callable[[Node], None]] = {
    "let_statement": let_statement,
    "assignment_statement": assignment_statement,
}
