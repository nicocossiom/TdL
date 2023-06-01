from tree_sitter import Node, Tree

from rule_functions import rule_functions


def ast_type_checker(tree: Tree):
    for child in tree.root_node.children:
        rule_functions[child.type](child)
