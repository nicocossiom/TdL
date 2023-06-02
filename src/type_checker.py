from typing import Any

from tree_sitter import Node, Tree

from errors import check_parsing_errors
from rule_functions import TypeCheckResult, rule_functions


def ast_type_check_tree(tree: Tree):
    check_parsing_errors(tree)
    for child in tree.root_node.named_children:
        rule_functions[child.type](child)


def ast_type_check_node(node: Node) -> TypeCheckResult | Any:
    return rule_functions[node.type](node)
