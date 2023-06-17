import json
from typing import Dict, List, Union

from tree_sitter import Node, Tree


def unwrap_text(text: bytes) -> str:
    decoded = text.decode("utf-8")
    decoded = decoded.strip('"')
    return decoded


def traverse_tree_named(tree: Tree):
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
    branches: set[int] = set()
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

        print(f"{prefix}{node.type} : {unwrap_text(node.text)}")


class ASTUtil():
    import logging
    LOGGER = logging.getLogger('ASTUtil')
    nested_dict_type = Dict[str, Union[str, List['nested_dict_type']]]
    # Simplify the AST

    def simplify_ast(self, tree: Tree, text: str):
        root = tree.root_node

        ignore_types = ["\n"]
        num_nodes = 0
        root_type = str(root.type)
        queue = [root]

        root_json: ASTUtil.nested_dict_type = {
            "node_type": root_type,
            "node_token": "",  # usually root does not contain token
            "children": []
        }

        queue_json = [root_json]
        while queue:

            current_node = queue.pop(0)
            current_node_json = queue_json.pop(0)
            num_nodes += 1

            for child in current_node.named_children:
                child_type = str(child.type)
                if child_type not in ignore_types:
                    queue.append(child)

                    child_token = ""
                    has_child = len(child.children) > 0

                    if not has_child:
                        child_token = text[child.start_byte:child.end_byte]
                    child_json: ASTUtil.nested_dict_type = {
                        "node_type": child_type,
                        "node_token": child_token,
                        "children": []
                    }

                    assert isinstance(current_node_json['children'], list)
                    children = current_node_json['children']
                    children.append(child_json)
                    queue_json.append(child_json)

        return root_json, num_nodes


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
