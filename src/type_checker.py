from tree_sitter import Tree

from rule_functions import rule_functions


def ast_type_check_tree(tree: Tree):
    for child in tree.root_node.named_children:
        rule_functions[child.type](child)
