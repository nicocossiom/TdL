from typing import Any, Callable

from tree_sitter import Node

from ast_util import unwrap_text
from errors import (NonInitializedError, PreDeclarationError,
                    TypeMismatchError, UndeclaredVariableError, print_error)
from language import language
from symbol_table import (FnEntry, JSPDLType, VarEntry, current_symbol_table,
                          symbol_table)


class TypeCheckResult():
    def __init__(self, type: JSPDLType, value: str | int | bool | None = None):
        self.type = type
        self.value = value

    def __repr__(self) -> str:
        return f"type: {self.type}, value: {self.value}"


def check_left_right_type_eq(
    left: TypeCheckResult,
    right: TypeCheckResult,
    node_left: Node,
    node_right: Node,
    wanted_type: JSPDLType
) -> bool:
    if left.type != right.type:
        if left.type == wanted_type:
            node = node_right
        else:
            node = node_left
        print_error(TypeMismatchError(node, left.type, right.type))
        return False
    if left.type != wanted_type:
        print_error(TypeMismatchError(node_left, wanted_type, left.type))
        return False
    return True


def let_statement(node: Node) -> TypeCheckResult | Any:
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


def assignment_statement(node: Node) -> TypeCheckResult | Any:
    from type_checker import ast_type_check_node
    query = language.query(
        "(assignment_statement (identifier) @identifier ( _ ) @expression)")
    captures = query.captures(node)
    identifier = unwrap_text(captures[0][0].text)
    expression = captures[1][0]

    if identifier not in current_symbol_table or identifier not in symbol_table:
        print_error(UndeclaredVariableError(node))
    else:
        var = symbol_table[identifier]
        if isinstance(var, FnEntry):
            print_error(TypeMismatchError(node, JSPDLType.FUNCTION, var.type))
            return
        expression_checked = ast_type_check_node(expression)
        check_left_right_type_eq(TypeCheckResult(
            var.type), expression_checked, var.node, expression, var.type)
        var.value = expression_checked.value


def value_to_typed_value(node: Node) -> int | str | bool:
    if node.type == "literal_string":
        return unwrap_text(node.text)
    elif node.type == "literal_number":
        return int(unwrap_text(node.text))
    elif node.type == "literal_boolean":
        return True if unwrap_text(node.text) == "true" else False
    else:
        raise Exception(f"Unknown value type {node.type}")


def post_increment_statement(node: Node) -> TypeCheckResult | Any:
    identifier = unwrap_text(node.text)
    if identifier not in current_symbol_table and identifier not in symbol_table:
        print_error(UndeclaredVariableError(node))
        return TypeCheckResult(JSPDLType.INVALID, None)
    var = current_symbol_table[identifier]

    if isinstance(var, FnEntry):
        print_error(TypeMismatchError(node, JSPDLType.FUNCTION, var.type))
        return TypeCheckResult(JSPDLType.INVALID, None)

    if not var.value:
        print_error(NonInitializedError(node))
        return TypeCheckResult(var.type, None)

    if not isinstance(var.value, int):
        raise Exception(f"this {var} value should be an int")
    var.value += 1
    return TypeCheckResult(JSPDLType.INT, var.value)


def get_id_from_symbol_table(identifier: str, node: Node):
    if identifier not in current_symbol_table and identifier not in symbol_table:
        print_error(UndeclaredVariableError(node))
        return TypeCheckResult(JSPDLType.INVALID, None)
    var = current_symbol_table[identifier]
    if isinstance(var, FnEntry):
        print_error(TypeMismatchError(
            node, JSPDLType.FUNCTION, var.type))
        return TypeCheckResult(JSPDLType.INVALID, None)
    if not var.value:
        print_error(NonInitializedError(node))
        return TypeCheckResult(JSPDLType.INVALID, None)
    return TypeCheckResult(var.type, var.value)


def value(node: Node) -> TypeCheckResult | Any:
    # $.post_increment_statement,
    # $.literal_string,
    # $.literal_number,
    # $.literal_boolean
    # this are the possible queries
    node = node.children[0]
    match node.type:
        case "post_increment_statement":
            return post_increment_statement(node.named_children[0])
        case "literal_string":
            return TypeCheckResult(JSPDLType.STRING, value_to_typed_value(node))
        case "literal_number":
            return TypeCheckResult(JSPDLType.INT, value_to_typed_value(node))
        case "literal_boolean":
            return TypeCheckResult(JSPDLType.BOOLEAN, value_to_typed_value(node))
        case "identifier":
            identifier = unwrap_text(node.text)
            return get_id_from_symbol_table(identifier, node)

        case _:
            raise Exception(f"Unknown value type {node.type}")


def or_expression(node: Node) -> TypeCheckResult | Any:
    node_left = node.named_children[0]
    node_right = node.named_children[1]
    left = rule_functions[node_left.type](node_left)
    right = rule_functions[node_right.type](node_right)
    if check_left_right_type_eq(left, right, node_left, node_right, left.type):
        return TypeCheckResult(JSPDLType.BOOLEAN, None)
    else:
        return TypeCheckResult(JSPDLType.INVALID, None)


def equality_expression(node: Node) -> TypeCheckResult | Any:
    node_left = node.named_children[0]
    node_right = node.named_children[1]
    left = rule_functions[node_left.type](node_left)
    right = rule_functions[node_right.type](node_right)
    if check_left_right_type_eq(left, right, node_left, node_right, left.type):
        return TypeCheckResult(JSPDLType.BOOLEAN, None)
    else:
        return TypeCheckResult(JSPDLType.INVALID, None)


def addition_expression(node: Node) -> TypeCheckResult | Any:
    base_query = language.query(
        "(addition_expression ( value ) @left ( value ) @right)")
    captures = base_query.captures(node)
    # base case 2 values with no more inner additions
    if (len(captures) == 2 and node.named_children[0].type != "addition_expression"):
        node_left = captures[0][0]
        node_right = captures[1][0]
        left = value(node_left)
        right = value(node_right)
        if check_left_right_type_eq(left, right, node_left, node_right, JSPDLType.INT):
            return TypeCheckResult(JSPDLType.INT, None)
        else:
            return TypeCheckResult(JSPDLType.INVALID, None)

    node_left = node.named_children[0]
    node_right = node.named_children[1]
    left = addition_expression(node_left)
    right = value(node_right)
    if check_left_right_type_eq(left, right, node_left, node_right, JSPDLType.INT):
        return TypeCheckResult(JSPDLType.INT, None)
    else:
        return TypeCheckResult(JSPDLType.INVALID, None)

    # return TypeCheckResult(JSPDLType.INT, None)
    #     left_checked = value(left)
    #     right_checked = value(right)
    #     if left_checked.type != right_checked.type:
    #         print_error(TypeMismatchError(
    #             node, left_checked.type, right_checked.type))
    #         return TypeCheckResult(JSPDLType.INVALID, None)
    #     assert isinstance(left_checked.value, int) and isinstance(
    #         right_checked.value, int)
    # return TypeCheckResult(JSPDLType.INT, None)


rule_functions: dict[str, Callable[[Node], Any | TypeCheckResult]] = {
    "let_statement": let_statement,
    "assignment_statement": assignment_statement,
    "value": value,
    "or_expression": or_expression,
    "equality_expression": equality_expression,
    "addition_expression": addition_expression,
    "post_increment_statement": post_increment_statement,
}
