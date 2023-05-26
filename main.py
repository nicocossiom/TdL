
from tree_sitter import Language, Node, Parser, Tree

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


def traverse_tree(tree: Tree):
    cursor = tree.walk()
    reached_root = False
    depth = 0

    while not reached_root:
        yield cursor.node, depth

        if cursor.goto_first_child():
            depth += 1
            continue

        while not cursor.goto_next_sibling():
            if not cursor.goto_parent():
                reached_root = True
                break
            depth -= 1


def traverse_tree_2(tree: Tree):

    def traverse_tree(node: Node):
        for n in node.children:
            if n.type == 'missing':
                print(f"missing {n.sexp()}")
            if n.type == 'error':
                print(f"error in {n.text}")
                print(n.sexp())
            else:
                print(n.type)
                match n.type:
                    case 'function_declaration':
                        print(n.sexp())
                        traverse_tree(n)
                    case other:
                        print(f"unrecognized {other}")

    traverse_tree(tree.root_node)


def print_tree(tree: Tree):
    for node, depth in traverse_tree(tree):
        print('  ' * depth + node.sexp())


def get_missing_nodes(tree: Tree) -> tuple[list[Node], list[Node]]:
    missing_nodes: list[Node] = []
    error_nodes: list[Node] = []

    def traverse_tree(node: Node):
        for n in node.children:
            if n.is_missing:
                missing_nodes.append(n)
            if n.has_error:
                print(f"error in {n.text}")
                print(n.sexp())
                error_nodes.append(n)

            traverse_tree(n)

    traverse_tree(tree.root_node)
    return missing_nodes, error_nodes


if __name__ == "__main__":
    with open('jspdl_language/test2.jspdl', 'r') as f:
        code = f.read()

    tree = parser.parse(bytes(code, 'utf8'))
    # print_tree(tree)
    traverse_tree_2(tree)
