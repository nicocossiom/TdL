import inspect
from typing import Any, List, Optional

from tree_sitter import Node, Tree

import code_gen as cg
import symbol_table as st
from ast_util import unwrap_text
from code_gen import Operand, Operation, Quartet
from errors import (CallWithInvalidArgumentsError, InvalidArgumentError,
                    InvalidReturnInScopeError, NonInitializedError,
                    PreDeclarationError, ReturnTypeMismatchError,
                    TypeMismatchError, UndeclaredFunctionCallError,
                    UndeclaredVariableError, print_error)
from language import language
from symbol_table import (Argument, DefinedFomOperation, FnEntry, JSPDLType,
                          SymbolTable, Undefined, VarEntry, size_dict)


def get_scope(identifier: str) -> cg.OperandScope:
    if identifier in st.current_symbol_table:
        # it's possible identifier is in both STs, in that case we want the local one to take precedence
        # it's in both STs and we're in the global one
        if identifier in st.global_symbol_table and st.global_symbol_table == st.current_symbol_table:
            return cg.OperandScope.GLOBAL
        else:
            return cg.OperandScope.LOCAL
    raise Exception(
        "Identifier not found in any symbol table, should not happen")


def ast_type_check_tree(tree: Tree):
    type_check_result = True
    for child in tree.root_node.named_children:
        if rule_functions[child.type](child).type == JSPDLType.INVALID:
            type_check_result = False
    return type_check_result


current_fn: Optional[FnEntry] = None


class TypeCheckResult():
    def __init__(self,
                 type: JSPDLType,
                 value:  cg.OpVal | None = None,
                 identifier: str | None = None,
                 offset: int | None = None,
                 scope: cg.OperandScope | None = None,
                 c3d_rep: str | None = None,
                 code_gen_rep: str | None = None
                 ):
        self.type = type
        self.value = value
        self.c3d_rep = c3d_rep
        self.code_gen_rep = code_gen_rep
        self.identifier = identifier
        self.offset = offset
        self.scope = get_scope(identifier) if identifier is not None else scope

    def __repr__(self) -> str:
        rep = "("
        for i, key in enumerate(self.__dict__, 0):
            if self.__dict__[key] is not None:
                rep += f"{key}: self.__dict__[val]" + \
                    ", " if i < len(self.__dict__) else ""
        rep += ")"
        return rep


def check_left_right_type_eq(
    left: TypeCheckResult,
    right: TypeCheckResult,
    node_left: Node,
    node_right: Node,
    wanted_type: JSPDLType | list[JSPDLType],
) -> bool:
    if left.type != right.type:
        if left.type == wanted_type:
            node = node_right
        else:
            node = node_left
        print_error(TypeMismatchError(node, [left.type], right.type))
        return False
    if left.type != wanted_type:
        if not isinstance(wanted_type, list):
            wanted_type = [wanted_type]
        print_error(TypeMismatchError(node_left, wanted_type, left.type))
        return False
    return True


def let_statement(node: Node):
    query = language.query(
        "(let_statement (type) @type (identifier) @identifier)")
    captures = query.captures(node)
    let_type = unwrap_text(captures[0][0].text)
    let_identifier = unwrap_text(captures[1][0].text)
    if let_identifier not in st.current_symbol_table:
        offset = size_dict[JSPDLType(let_type)]
        st.current_symbol_table[let_identifier] = VarEntry(
            type=let_type, value=Undefined(), offset=offset, node=node)
        cg.static_memory_size += offset
        return TypeCheckResult(JSPDLType.VOID)
    else:
        pred_node = st.current_symbol_table[let_identifier]
        assert isinstance(pred_node, VarEntry)
        print_error(PreDeclarationError(node, pred_node.node))
        return TypeCheckResult(JSPDLType.INVALID)


def assignment_statement(node: Node):
    query = language.query(
        "(assignment_statement (identifier) @identifier ( _ ) @expression)")
    captures = query.captures(node)
    identifier = unwrap_text(captures[0][0].text)
    expression = captures[1][0]

    if identifier not in st.current_symbol_table and identifier not in st.global_symbol_table:
        print_error(UndeclaredVariableError(node))
        return TypeCheckResult(JSPDLType.INVALID)
    var = st.current_symbol_table[identifier]

    expression_checked: TypeCheckResult = rule_functions[expression.type](
        expression)
    if isinstance(var, FnEntry):
        print_error(TypeMismatchError(
            node, expression_checked.type, var.type))
        return TypeCheckResult(JSPDLType.INVALID)
    if not check_left_right_type_eq(TypeCheckResult(
            var.type), expression_checked, var.node, expression, var.type):
        return TypeCheckResult(JSPDLType.INVALID)
    # Assignment is correct
    scope = get_scope(identifier)
    assert isinstance(var, VarEntry)
    var.value = DefinedFomOperation()
    assert isinstance(var, VarEntry)
    # TODO check if var.offset is the one
    cg.c3d_queue.append(
        f"{identifier} := {expression_checked.c3d_rep}")
    cg.quartet_queue.append(
        Quartet(
            Operation.ASSIGN,
            op1=Operand(offset=var.offset, scope=scope),
            res=Operand(value=expression_checked.value, offset=expression_checked.offset, scope=expression_checked.scope)))
    return TypeCheckResult(JSPDLType.VOID)


def get_value_as_str_from_node(node: Node) -> str:
    if node.type == "literal_boolean":
        return "1" if unwrap_text(node.text) == "true" else "0"
    if node.type in ["literal_number", "literal_string"]:
        return unwrap_text(node.text)
    else:
        raise Exception(f"Unknown value type {node.type}")


def post_increment_statement(node: Node) -> TypeCheckResult | Any:
    node = node.children[0]
    identifier = unwrap_text(node.text)
    if identifier not in st.current_symbol_table and identifier not in st.global_symbol_table:
        print_error(UndeclaredVariableError(node))
        return TypeCheckResult(JSPDLType.INVALID)
    var = st.current_symbol_table[identifier]

    if isinstance(var, FnEntry):
        print_error(TypeMismatchError(node, JSPDLType.FUNCTION, var.type))
        return TypeCheckResult(JSPDLType.INVALID)
    assert isinstance(var, VarEntry)
    if not var.value:
        print_error(NonInitializedError(node))
        return TypeCheckResult(var.type)
    scope = get_scope(identifier)
    cg.c3d_queue.append(f"{identifier} := {identifier} + 1")
    cg.quartet_queue.append(
        Quartet(Operation.INC,
                Operand(offset=var.offset, scope=scope),
                )
    )
    return TypeCheckResult(JSPDLType.INT, identifier=identifier, offset=var.offset, scope=scope)


def get_trs_from_ts_with_id(identifier: str, node: Node):
    if identifier not in st.current_symbol_table and identifier not in st.global_symbol_table:
        print_error(UndeclaredVariableError(node))
        return TypeCheckResult(JSPDLType.INVALID)
    var = st.current_symbol_table[identifier] if identifier in st.current_symbol_table else st.global_symbol_table[identifier]
    if isinstance(var, FnEntry):
        print_error(TypeMismatchError(
            node, [JSPDLType.FUNCTION], var.type))
        return TypeCheckResult(JSPDLType.INVALID)
    assert isinstance(var, VarEntry)
    return TypeCheckResult(var.type, scope=get_scope(identifier), offset=var.offset, identifier=identifier)


def get_trs_from_ts_with_id_and_value(identifier: str, node: Node):
    try:
        var = st.current_symbol_table[identifier] if identifier in st.current_symbol_table else st.global_symbol_table[identifier]
        if isinstance(var, FnEntry):
            print_error(TypeMismatchError(
                node, [JSPDLType.FUNCTION], var.type))
            return TypeCheckResult(JSPDLType.INVALID)
        assert isinstance(var, VarEntry)
        if isinstance(var.value, Undefined):
            print_error(NonInitializedError(node))
            return TypeCheckResult(JSPDLType.INVALID)
        return TypeCheckResult(
            type=var.type, identifier=identifier, offset=var.offset,
            c3d_rep=identifier, scope=get_scope(identifier))
    except KeyError:
        print_error(UndeclaredVariableError(node))
        return TypeCheckResult(JSPDLType.INVALID)


def value(node: Node) -> TypeCheckResult:
    # $.post_increment_statement,
    # $.literal_string,
    # $.literal_number,
    # $.literal_boolean
    # this are the possible queries
    node = node.children[0]
    match node.type:
        case "post_increment_statement":
            return post_increment_statement(node)
        case "identifier":
            identifier = unwrap_text(node.text)
            return get_trs_from_ts_with_id_and_value(identifier, node)
        case "literal_string":
            literal_val = get_value_as_str_from_node(node)
            return TypeCheckResult(JSPDLType.STRING, cg.OpVal(literal_val), c3d_rep=literal_val)
        case "literal_number":
            literal_val = get_value_as_str_from_node(node)
            # 3cd representation for inmeadite value is #value
            return TypeCheckResult(JSPDLType.INT, cg.OpVal(literal_val), c3d_rep=f"#{literal_val}")
        case "literal_boolean":
            # booleans are represented as 1 and 0 hence they are literal numbers
            # 3cd representation for inmeadite value is #value
            literal_val = get_value_as_str_from_node(node)
            return TypeCheckResult(JSPDLType.BOOLEAN, cg.OpVal(literal_val), c3d_rep=f"#{literal_val}")
        case _:
            raise Exception(f"Unknown value type {node.type}")


def or_expression(node: Node) -> TypeCheckResult:
    node_left = node.named_children[0]
    node_right = node.named_children[1]
    left = rule_functions[node_left.type](node_left)
    right = rule_functions[node_right.type](node_right)
    if check_left_right_type_eq(left, right, node_left, node_right, left.type):
        cg.c3d_queue.append(
            f"t{cg.temporal_counter} := {id_if_not_literal_value(left)} || {id_if_not_literal_value(right)}")
        cg.temporal_counter += 1
        res_op_val = cg.OpVal(rep=cg.OpValRep(cg.OpValRepType.ACCUMULATOR))
        cg.quartet_queue.append(
            Quartet(Operation.OR,
                    Operand(left.value, left.offset, left.scope),
                    Operand(right.value, right.offset, right.scope),
                    res=Operand(value=res_op_val,
                                scope=cg.OperandScope.TEMPORAL)
                    )
        )
        return TypeCheckResult(JSPDLType.BOOLEAN, c3d_rep=f"t{cg.temporal_counter-1}", code_gen_rep=".A", scope=cg.OperandScope.TEMPORAL)
    else:
        return TypeCheckResult(JSPDLType.INVALID)


def equality_expression(node: Node) -> TypeCheckResult:
    node_left = node.named_children[0]
    node_right = node.named_children[1]
    left = rule_functions[node_left.type](node_left)
    right = rule_functions[node_right.type](node_right)
    if check_left_right_type_eq(left, right, node_left, node_right, left.type):
        cg.c3d_queue.append(
            f"""if {id_if_not_literal_value(left)} == {id_if_not_literal_value(right)} goto true
t{cg.temporal_counter} := 0
goto next
true: 
t{cg.temporal_counter} := 1
next:
"""
        )
        res_op_val = cg.OpVal(rep=cg.OpValRep(cg.OpValRepType.REGISTER, ".R2"))
        cg.quartet_queue.append(
            Quartet(Operation.EQUALS,
                    Operand(left.value, left.offset, left.scope),
                    Operand(right.value, right.offset, right.scope),
                    res=Operand(value=res_op_val,
                                scope=cg.OperandScope.TEMPORAL)
                    )
        )
        cg.temporal_counter += 1
        return TypeCheckResult(
            JSPDLType.BOOLEAN,
            value=res_op_val,
            c3d_rep=f"t{cg.temporal_counter}",
            code_gen_rep=".R2",
            scope=cg.OperandScope.TEMPORAL
        )
    else:
        return TypeCheckResult(JSPDLType.INVALID)


def id_if_not_literal_value(x: TypeCheckResult):
    return x.value if not x.identifier else x.identifier


def addition_expression(node: Node) -> TypeCheckResult:
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
            cg.c3d_queue.append("t" + str(cg.temporal_counter) +
                                " := " + str(left.value) + " + " + str(right.value))
            cg.temporal_counter += 1
            res_op_val = cg.OpVal(rep=cg.OpValRep(
                cg.OpValRepType.REGISTER, ".A"))
            cg.quartet_queue.append(
                Quartet(
                    Operation.ADD,
                    Operand(value=left.value,
                            offset=left.offset, scope=left.scope),
                    Operand(value=right.value,
                            offset=right.offset, scope=right.scope),
                    Operand(value=res_op_val,  scope=cg.OperandScope.TEMPORAL)
                )
            )
            return TypeCheckResult(JSPDLType.INT, value=res_op_val, c3d_rep=f"t{+ cg.temporal_counter - 1}", code_gen_rep=".A", scope=cg.OperandScope.TEMPORAL)
        else:
            return TypeCheckResult(JSPDLType.INVALID)

    node_left = node.named_children[0]
    node_right = node.named_children[1]
    left = addition_expression(node_left)
    right = value(node_right)
    if check_left_right_type_eq(left, right, node_left, node_right, JSPDLType.INT):
        cg.c3d_queue.append(
            f"t{cg.temporal_counter} := {left.c3d_rep} + {id_if_not_literal_value(right)}")
        cg.temporal_counter += 1
        res_op_val = cg.OpVal(rep=cg.OpValRep(
            cg.OpValRepType.REGISTER, ".A"))
        cg.quartet_queue.append(
            Quartet(
                Operation.ADD,
                Operand(value=left.value, scope=cg.OperandScope.TEMPORAL),
                Operand(value=right.value, offset=right.offset,
                        scope=right.scope),
                Operand(value=res_op_val, scope=cg.OperandScope.TEMPORAL)
            )
        )
        return TypeCheckResult(JSPDLType.INT, value=res_op_val, c3d_rep=f"t{cg.temporal_counter}", code_gen_rep=".A", scope=cg.OperandScope.TEMPORAL)
    else:
        return TypeCheckResult(JSPDLType.INVALID)


def input_statement(node: Node) -> TypeCheckResult:
    identifier_node = node.named_children[0]
    identifier = unwrap_text(identifier_node.text)
    trs = get_trs_from_ts_with_id(identifier, identifier_node)
    expected_types = [JSPDLType.INT, JSPDLType.STRING]
    if (check_left_right_type_eq(trs, TypeCheckResult(JSPDLType.STRING), identifier_node, identifier_node, expected_types)) \
            or \
            (check_left_right_type_eq(trs, TypeCheckResult(JSPDLType.INT), identifier_node, identifier_node, expected_types)):
        cg.quartet_queue.append(
            Quartet(
                Operation.INPUT,
                Operand(value=trs.value, offset=trs.offset,
                        op_type=trs.type, scope=trs.scope),
            )
        )
        return TypeCheckResult(trs.type, scope=trs.scope, identifier=identifier)
    else:
        return TypeCheckResult(JSPDLType.INVALID)


def print_statement(node: Node):
    expression = node.named_children[0]
    expres_checked = rule_functions[expression.type](expression)
    if expres_checked.type not in [JSPDLType.INT, JSPDLType.STRING, JSPDLType.BOOLEAN]:
        print_error(InvalidArgumentError(node))
        return TypeCheckResult(JSPDLType.INVALID)


def return_statement(node: Node) -> TypeCheckResult:
    if (not current_fn):
        print_error(InvalidReturnInScopeError(node))
        return TypeCheckResult(JSPDLType.INVALID)
    query = language.query("(return_statement ( value ) @value)")
    captures = query.captures(node)
    res = TypeCheckResult(JSPDLType.VOID) if (
        len(captures) == 0) else rule_functions[captures[0][0].type](captures[0][0])
    if (res.type != current_fn.return_type):
        print_error(ReturnTypeMismatchError(
            node, current_fn, res.type))
        return TypeCheckResult(JSPDLType.INVALID)
    return res


def function_declaration(node: Node) -> TypeCheckResult:
    # TODO terminar y chequear
    query = language.query(
        """
    (function_declaration
      (identifier) @identifier
      (type)? @type
      (argument_declaration_list) @argument_declaration_list
      (block) @block
    )
    """
    )

    captures = query.captures(node)
    capt_dict = {capture[1]: capture[0] for capture in captures}
    identifier = unwrap_text(capt_dict["identifier"].text)
    if identifier in st.global_symbol_table:
        print_error(PreDeclarationError(
            capt_dict["identifier"], st.global_symbol_table[identifier].node))
        return TypeCheckResult(JSPDLType.INVALID)
    ret_type = JSPDLType(capt_dict["type"]
                         ) if "type" in capt_dict else JSPDLType.VOID
    args = argument_declaration_list(
        capt_dict["argument_declaration_list"]) if "argument_declaration_list" in capt_dict else []
    if "block" not in capt_dict:
        raise Exception("Block not found in function declaration")
    st.current_symbol_table = SymbolTable()
    global current_fn
    current_fn = FnEntry(ret_type, args, node)
    block_check = rule_functions["block"](capt_dict["block"])
    if block_check.type == JSPDLType.INVALID:
        return TypeCheckResult(JSPDLType.INVALID)
    st.global_symbol_table[identifier] = FnEntry(ret_type, args, node)
    st.current_symbol_table = st.global_symbol_table
    return TypeCheckResult(type=JSPDLType.VOID, identifier=identifier)


def block(node: Node) -> TypeCheckResult:
    # TODO terminar y chequear
    for node in node.named_children:
        if (rule_functions[node.type](node).type == JSPDLType.INVALID):
            return TypeCheckResult(JSPDLType.INVALID)
    return TypeCheckResult(JSPDLType.VOID)


def argument_declaration_list(node: Node) -> list[Argument]:
    arg_list: list[Argument] = []
    query = language.query(
        "(argument_declaration_list (argument_declaration) @argument_declaration)"
    )
    captures = query.captures(node)
    if len(captures) == 0:
        return arg_list
    for argument in captures:
        arg_list.append(argument_declaration(argument[0]))
    return arg_list


def argument_declaration(node: Node) -> Argument:
    query = language.query(
        "(argument_declaration ( type ) @type ( identifier ) @identifier)"
    )
    captures = query.captures(node)
    capt_dict: dict[str, Node] = {capture[1]: capture[0]
                                  for capture in captures}
    return Argument(JSPDLType(unwrap_text(capt_dict["type"].text)), unwrap_text(capt_dict["identifier"].text))


def function_call(node: Node) -> TypeCheckResult:
    # TODO test
    query = language.query(
        "(function_call ( identifier ) @identifier ( argument_list)? @argument_list)")
    captures = query.captures(node)
    identifier = unwrap_text(captures[0][0].text)
    if identifier not in st.current_symbol_table or identifier not in st.global_symbol_table:
        print_error(UndeclaredFunctionCallError(captures[0][0]))
        return TypeCheckResult(JSPDLType.INVALID)
    fn = st.current_symbol_table[identifier] if identifier in st.current_symbol_table else st.global_symbol_table[identifier]
    assert isinstance(fn, FnEntry)
    fn_args = [arg.type for arg in fn.arguments]
    args = argument_list(captures[1][0])
    if args is None:
        print_error(CallWithInvalidArgumentsError(
            node, fn_args, [JSPDLType.INVALID]))
        return TypeCheckResult(JSPDLType.INVALID)
    if args != fn_args:
        print_error(CallWithInvalidArgumentsError(node, fn_args, args))
        return TypeCheckResult(JSPDLType.INVALID)
    return TypeCheckResult(fn.return_type)


def argument_list(node: Node) -> List[JSPDLType] | None:
    # TODO test
    arg_list: list[JSPDLType] = []
    for val in node.named_children:
        val_checked = rule_functions[val.type](val)
        if val_checked.type == JSPDLType.INVALID:
            return None
        arg_list.append(val_checked.type)
    return arg_list


# Get the current module
current_module = inspect.getmodule(lambda: None)

# Retrieve all functions in the current module
rule_functions = {name: func for name, func in inspect.getmembers(
    current_module, inspect.isfunction)}

# rule_functions: dict[str, Callable[[Node], TypeCheckResult | list[Argument] | List[JSPDLType] | None]] = {
#     "let_statement": let_statement,
#     "assignment_statement": assignment_statement,
#     "value": value,
#     "or_expression": or_expression,
#     "equality_expression": equality_expression,
#     "addition_expression": addition_expression,
#     "post_increment_statement": post_increment_statement,
#     "input_statement": input_statement,
#     "print_statement": print_statement,
#     "return_statement": return_statement,
#     "function_call": function_call,
#     "argument_listt": argument_list,
#     "function_declaration": function_declaration,
#     "block": block,
#     "argument_declaration_list": argument_declaration_list,
#     "argument_declaration": argument_declaration,
# }
