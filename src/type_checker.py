import inspect
from typing import Any, List, Optional

from tree_sitter import Node, Tree

import code_gen as cg
import symbol_table as st
from ast_util import unwrap_text
from code_gen import Operand, Operation, Quartet
from errors import (CallWithInvalidArgumentsError, InvalidArgumentError,
                    NonInitializedError, NoValidReturnInFunctionBody,
                    PreDeclarationError, ReturnTypeMismatchError,
                    TypeMismatchError, UncallableError,
                    UndeclaredFunctionCallError, UndeclaredVariableError,
                    print_error)
from language import language
from symbol_table import (Argument, DefinedFomOperation, FnEntry, JSPDLType,
                          Undefined, VarEntry, VarEntryValType)


def get_scope(identifier: str) -> cg.OperandScope:
    if identifier in st.current_symbol_table:
        # it's possible identifier is in both STs, in that case we want the local one to take precedence
        # it's in both STs and we're in the global one
        if identifier in st.global_symbol_table and st.global_symbol_table == st.current_symbol_table:
            return cg.OperandScope.GLOBAL
        else:
            return cg.OperandScope.LOCAL
    elif identifier in st.global_symbol_table:
        return cg.OperandScope.GLOBAL
    else:
        raise Exception(
            "Identifier not found in any symbol table, should not happen")


def ast_type_check_tree(tree: Tree):
    st.global_symbol_table["__main__"] = FnEntry(
        JSPDLType.VOID, [], tree.root_node)
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
                 ):
        self.type = type
        self.value = value
        self.c3d_rep = c3d_rep
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
    expected_types: list[JSPDLType],
) -> bool:
    if left.type != right.type:
        if left.type not in expected_types:
            node = node_right
        else:
            node = node_left
        print_error(TypeMismatchError(node, [left.type], right.type))
        return False
    if left.type not in expected_types:
        print_error(TypeMismatchError(node_left, expected_types, left.type))
        return False
    return True


def let_statement(node: Node):
    query = language.query(
        "(let_statement (type) @type (identifier) @identifier)")
    captures = query.captures(node)
    let_type = unwrap_text(captures[0][0].text)
    let_identifier = unwrap_text(captures[1][0].text)
    if let_identifier not in st.current_symbol_table:
        st.current_symbol_table[let_identifier] = VarEntry(
            type=let_type, value=Undefined(), node=node)
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
    if identifier in st.current_symbol_table and st.current_symbol_table != st.global_symbol_table:
        var = st.current_symbol_table[identifier]
        scope = cg.OperandScope.LOCAL
        ar_offset = st.current_symbol_table.access_register_size
    else:
        var = st.global_symbol_table[identifier]
        scope = cg.OperandScope.GLOBAL
        ar_offset = 0

    expression_checked: TypeCheckResult = rule_functions[expression.type](
        expression)
    if isinstance(var, FnEntry):
        print_error(TypeMismatchError(
            node, expression_checked.type, var.type))
        return TypeCheckResult(JSPDLType.INVALID)
    if not check_left_right_type_eq(TypeCheckResult(
            var.type), expression_checked, captures[0][0], expression, [var.type]):
        return TypeCheckResult(JSPDLType.INVALID)
    assert isinstance(var, VarEntry)
    var.value = DefinedFomOperation()
    assert isinstance(var, VarEntry)
    cg.c3d_queue.append(
        f"{identifier} := {expression_checked.c3d_rep}")
    cg.quartet_queue.append(
        Quartet(
            Operation.ASSIGN,
            op1=Operand(offset=var.offset, op_type=var.type, scope=scope),
            res=Operand(value=expression_checked.value,
                        offset=expression_checked.offset, scope=expression_checked.scope),
            op_options={"ar_offset": ar_offset}
        ),
    )
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
    var = st.current_symbol_table[identifier] \
        if identifier in st.current_symbol_table \
        else st.global_symbol_table[identifier]

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


def get_trs_from_ts_with_id(identifier: str, node: Node, modify: VarEntryValType | None = None):
    if identifier not in st.current_symbol_table and identifier not in st.global_symbol_table:
        print_error(UndeclaredVariableError(node))
        return TypeCheckResult(JSPDLType.INVALID)
    table_origin = None
    if identifier in st.current_symbol_table:
        var = st.current_symbol_table[identifier]
        table_origin = st.current_symbol_table
    else:
        var = st.global_symbol_table[identifier]
        table_origin = st.global_symbol_table
    if isinstance(var, FnEntry):
        print_error(TypeMismatchError(
            node, [JSPDLType.FUNCTION], var.type))
        return TypeCheckResult(JSPDLType.INVALID)
    assert isinstance(var, VarEntry)
    if modify:
        table_origin[identifier].value = modify  # type: ignore
    var.value
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
    node = node.children[0]
    match node.type:
        case "post_increment_statement":
            return post_increment_statement(node)
        case "identifier":
            identifier = unwrap_text(node.text)
            return get_trs_from_ts_with_id_and_value(identifier, node)
        case "literal_string":
            literal_val = get_value_as_str_from_node(node)
            return TypeCheckResult(
                JSPDLType.STRING,
                cg.OpVal(rep=cg.OpValRep(
                    rep_value=literal_val,
                    rep_type=cg.OpValRepType.LITERAL
                )),
                c3d_rep=literal_val, scope=cg.OperandScope.TEMPORAL
            )
        case "literal_number":
            literal_val = get_value_as_str_from_node(node)
            return TypeCheckResult(
                JSPDLType.INT,
                cg.OpVal(rep=cg.OpValRep(
                    rep_value=f"#{literal_val}",
                    rep_type=cg.OpValRepType.LITERAL
                )),
                c3d_rep=literal_val, scope=cg.OperandScope.TEMPORAL
            )
        case "literal_boolean":
            literal_val = get_value_as_str_from_node(node)
            return TypeCheckResult(
                JSPDLType.BOOLEAN,
                cg.OpVal(rep=cg.OpValRep(
                    rep_value=f"#{literal_val}",
                    rep_type=cg.OpValRepType.LITERAL
                )),
                c3d_rep=literal_val, scope=cg.OperandScope.TEMPORAL
            )
        case "function_call":
            return function_call(node)
        case _:
            raise Exception(f"Unknown value type {node.type}")


def or_expression(node: Node) -> TypeCheckResult:
    node_left = node.named_children[0]
    node_right = node.named_children[1]
    left: TypeCheckResult = rule_functions[node_left.type](node_left)
    right: TypeCheckResult = rule_functions[node_right.type](node_right)
    if check_left_right_type_eq(left, right, node_left, node_right, [left.type]):
        cg.c3d_queue.append(
            f"{cg.get_new_temporal_per_st()} := {left.c3d_rep} || {right.c3d_rep}")
        res_op_val = cg.OpVal(rep=cg.OpValRep(cg.OpValRepType.ACCUMULATOR))
        cg.quartet_queue.append(
            Quartet(Operation.OR,
                    Operand(left.value, left.offset, left.scope),
                    Operand(right.value, right.offset, right.scope),
                    res=Operand(value=res_op_val,
                                scope=cg.OperandScope.TEMPORAL)
                    )
        )
        return TypeCheckResult(JSPDLType.BOOLEAN, value=res_op_val, c3d_rep=cg.get_last_temporal_st(), scope=cg.OperandScope.TEMPORAL)
    else:
        return TypeCheckResult(JSPDLType.INVALID)


def equality_expression(node: Node) -> TypeCheckResult:
    node_left = node.named_children[0]
    node_right = node.named_children[1]
    left: TypeCheckResult = rule_functions[node_left.type](node_left)
    right: TypeCheckResult = rule_functions[node_right.type](node_right)
    if check_left_right_type_eq(left, right, node_left, node_right, [left.type]):
        cg.c3d_queue.append(
            f"""if {left.c3d_rep} == {right.c3d_rep} goto true_{cg.expression_tag_counter}
{cg.get_new_temporal_per_st()} := 0
goto next_{cg.expression_tag_counter}
true_{cg.expression_tag_counter}: 
{cg.get_last_temporal_st()} := 1
next_{cg.expression_tag_counter}:
"""
        )
        cg.expression_tag_counter += 1
        res_op_val = cg.OpVal(rep=cg.OpValRep(cg.OpValRepType.REGISTER, ".R2"))
        cg.quartet_queue.append(
            Quartet(Operation.EQUALS,
                    Operand(left.value, left.offset, left.scope),
                    Operand(right.value, right.offset, right.scope),
                    res=Operand(value=res_op_val,
                                scope=cg.OperandScope.TEMPORAL)
                    )
        )
        return TypeCheckResult(
            JSPDLType.BOOLEAN,
            value=res_op_val,
            c3d_rep=cg.get_new_temporal_per_st(),
            scope=cg.OperandScope.TEMPORAL
        )
    else:
        return TypeCheckResult(JSPDLType.INVALID)


def id_if_not_literal_value(x: TypeCheckResult):
    return x.value if not x.identifier else x.identifier


def addition_expression(node: Node) -> TypeCheckResult:
    result_offset = st.current_symbol_table.access_register_size if st.current_symbol_table != st.global_symbol_table else 0
    base_query = language.query(
        "(addition_expression ( value ) @left ( value ) @right)")
    captures = base_query.captures(node)
    # base case 2 values with no more inner additions
    if (len(captures) == 2 and node.named_children[0].type != "addition_expression"):
        node_left = captures[0][0]
        node_right = captures[1][0]
        left: TypeCheckResult = rule_functions[node_left.type](node_left)
        right: TypeCheckResult = rule_functions[node_right.type](node_right)
        if not check_left_right_type_eq(left, right, node_left, node_right, [JSPDLType.INT]):
            return TypeCheckResult(JSPDLType.INVALID)
        assert left.c3d_rep is not None and right.c3d_rep is not None
        cg.c3d_queue.append(cg.get_new_temporal_per_st() +
                            " := " + left.c3d_rep + " + " + right.c3d_rep)
        cg.quartet_queue.append(
            Quartet(
                Operation.ADD,
                Operand(value=left.value,
                        offset=left.offset, scope=left.scope),
                Operand(value=right.value,
                        offset=right.offset, scope=right.scope),
                Operand(offset=result_offset, scope=cg.OperandScope.LOCAL)
            )
        )
        return TypeCheckResult(JSPDLType.INT, offset=result_offset, c3d_rep=cg.get_last_temporal_st(), scope=cg.OperandScope.LOCAL)

    node_left = node.named_children[0]
    node_right = node.named_children[1]
    left = addition_expression(node_left)
    right = value(node_right)

    if not check_left_right_type_eq(left, right, node_left, node_right, [JSPDLType.INT]):
        return TypeCheckResult(JSPDLType.INVALID)

    cg.c3d_queue.append(
        f"{cg.get_new_temporal_per_st()} := {left.c3d_rep} + {id_if_not_literal_value(right)}")
    cg.quartet_queue.append(
        Quartet(
            Operation.ADD,
            Operand(value=left.value, scope=left.scope, offset=left.offset),
            Operand(value=right.value, offset=right.offset, scope=right.scope),
            Operand(offset=result_offset, scope=cg.OperandScope.LOCAL)
        )
    )
    return TypeCheckResult(JSPDLType.INT, offset=result_offset, c3d_rep=cg.get_last_temporal_st(), scope=cg.OperandScope.LOCAL)


def input_statement(node: Node) -> TypeCheckResult:
    identifier_node = node.named_children[0]
    trs: TypeCheckResult = get_trs_from_ts_with_id(
        unwrap_text(identifier_node.text), identifier_node, modify=DefinedFomOperation())
    if trs.type not in [JSPDLType.INT, JSPDLType.STRING]:
        print_error(InvalidArgumentError(node))
        return TypeCheckResult(JSPDLType.INVALID)
    cg.quartet_queue.append(
        Quartet(
            Operation.INPUT,
            Operand(value=trs.value, offset=trs.offset,
                    op_type=trs.type, scope=trs.scope),
        )
    )
    return TypeCheckResult(JSPDLType.VOID)


def print_statement(node: Node):
    expression = node.named_children[0]
    expres_checked: TypeCheckResult = rule_functions[expression.type](
        expression)
    if expres_checked.type not in [JSPDLType.INT, JSPDLType.STRING, JSPDLType.BOOLEAN]:
        print_error(InvalidArgumentError(node))
        return TypeCheckResult(JSPDLType.INVALID)
    if expres_checked.type == JSPDLType.STRING:
        if expres_checked.value is not None and expres_checked.value.rep is not None \
                and expres_checked.value.rep.rep_type == cg.OpValRepType.LITERAL:
            assert expres_checked.value.rep.rep_value is not None
            lit_n = len(cg.literals_queue)
            cg.literals_queue.append(
                cg.gen_instr(
                    f"lit{lit_n+1}: DATA \"{expres_checked.value.rep.rep_value}\"")
            )
            expres_checked.value.rep.rep_value = f"/lit{lit_n +1}"

    cg.quartet_queue.append(
        Quartet(
            Operation.PRINT,
            Operand(value=expres_checked.value, offset=expres_checked.offset,
                    op_type=expres_checked.type, scope=expres_checked.scope)
        )
    )
    return TypeCheckResult(JSPDLType.VOID)


def return_statement(node: Node) -> TypeCheckResult:
    query = language.query("(return_statement ( _ )? @value)")
    captures = query.captures(node)
    global current_fn
    options = {}
    if current_fn is not None:
        options = {
            "tag_identifier": current_fn.function_name,
            "access_register_size": st.current_symbol_table.access_register_size
        }
    if len(captures) == 0:
        cg.c3d_queue.append("return")
        cg.quartet_queue.append(
            Quartet(
                Operation.RETURN,
                op_options=options
            )
        )
        return TypeCheckResult(JSPDLType.VOID)
    # there is return value
    if st.current_symbol_table == st.global_symbol_table:
        # returning value from global scope not allowed
        main_fn = st.global_symbol_table["___main___"]
        assert isinstance(main_fn, st.FnEntry)
        print_error(ReturnTypeMismatchError(
            node, main_fn, JSPDLType.VOID))
        return TypeCheckResult(JSPDLType.INVALID)
    # returning value from local scope
    value_checked: TypeCheckResult = rule_functions[captures[0][0].type](
        captures[0][0])
    assert current_fn is not None
    if value_checked.type != current_fn.return_type:
        print_error(ReturnTypeMismatchError(
            node, current_fn, value_checked.type))
        return TypeCheckResult(JSPDLType.INVALID)
    # valid return value
    if (node.parent is not None and node.parent.parent is not None and node.parent.parent.type == "function_declaration"):
        # top level return -> function has at least one return that will always be executed
        current_fn.has_at_least_one_return = True
    cg.c3d_queue.append(f"return {value_checked.c3d_rep}")
    if options.get("access_register_size") is not None:
        assert isinstance(options["access_register_size"], int)
        options["access_register_size"] += st.size_dict[value_checked.type]
    cg.quartet_queue.append(
        Quartet(
            Operation.RETURN,
            Operand(value=value_checked.value, offset=value_checked.offset,
                    op_type=value_checked.type, scope=value_checked.scope),
            op_options=options
        )
    )
    return TypeCheckResult(value_checked.type)


def function_declaration(node: Node) -> TypeCheckResult:
    query = language.query(
        """
    (function_declaration
      (identifier) @identifier
      (type)? @type
      (argument_declaration_list) @argument_declaration_list
      (block_and_declaration) @block
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
    ret_type = JSPDLType(unwrap_text(capt_dict["type"].text)) if capt_dict.get(
        "type") is not None else JSPDLType.VOID
    st.current_symbol_table = st.SymbolTable(parent=st.global_symbol_table)
    args = argument_declaration_list(
        capt_dict["argument_declaration_list"]) if "argument_declaration_list" in capt_dict else []

    cg.functions_tag_counters[identifier] = 0
    cg.quartet_queue.append(
        Quartet(
            Operation.FUNCTION_DECLARATION_START_TAG,
            op_options={"tag_identifier": identifier}
        )
    )

    global current_fn
    current_fn = FnEntry(ret_type, args, node)
    st.global_symbol_table[identifier] = current_fn
    cg.c3d_queue.append(f"go to {identifier}_end")

    block_checked = rule_functions[capt_dict["block"].type](capt_dict["block"])
    if block_checked.type == JSPDLType.INVALID:
        return TypeCheckResult(JSPDLType.INVALID)
    if ret_type != JSPDLType.VOID and not current_fn.has_at_least_one_return:
        print_error(NoValidReturnInFunctionBody(capt_dict["identifier"]))
        return TypeCheckResult(JSPDLType.INVALID)

    cg.quartet_queue.append(
        Quartet(
            Operation.RETURN,
            op_options={"tag_identifier": identifier}
        )
    )
    cg.c3d_queue.append(f"{identifier}_end:")
    cg.quartet_queue.append(
        Quartet(
            Operation.FUNCTION_DECLARATION_END_TAG,
            op_options={"tag_identifier": identifier}
        )
    )
    current_fn = None
    st.current_symbol_table = st.global_symbol_table
    return TypeCheckResult(type=JSPDLType.VOID, identifier=identifier)


def block(node: Node) -> TypeCheckResult:
    for node in node.named_children:
        if (rule_functions[node.type](node).type == JSPDLType.INVALID):
            return TypeCheckResult(JSPDLType.INVALID)
    return TypeCheckResult(JSPDLType.VOID)


def block_and_declaration(node: Node) -> TypeCheckResult:
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
        arg_checked = argument_declaration(argument[0])
        st.current_symbol_table[arg_checked.id] = VarEntry(
            type=arg_checked.type.__str__(), node=argument[0], value=st.DefinedFomOperation())
        arg_list.append(arg_checked)
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
    query = language.query(
        "(function_call ( identifier ) @identifier ( argument_list)? @argument_list)")
    captures = query.captures(node)
    identifier = unwrap_text(captures[0][0].text)
    if identifier not in st.global_symbol_table:
        print_error(UndeclaredFunctionCallError(captures[0][0]))
        return TypeCheckResult(JSPDLType.INVALID)
    fn = st.global_symbol_table[identifier]
    if isinstance(fn, VarEntry):
        print_error(UncallableError(captures[0][0], fn.type))
        return TypeCheckResult(JSPDLType.INVALID)
    assert isinstance(fn, FnEntry)
    fn_args = [arg.type for arg in fn.arguments]
    if len(captures) == 1 and len(fn_args) != 0:
        # function call without arguments but function expects arguments
        print_error(CallWithInvalidArgumentsError(
            node, fn_args, []))
        return TypeCheckResult(JSPDLType.INVALID)
    elif len(captures) >= 2:
        # function call with arguments
        args = argument_list(captures[1][0])
        if args is None:
            print_error(CallWithInvalidArgumentsError(
                node, fn_args, [JSPDLType.INVALID]))
            return TypeCheckResult(JSPDLType.INVALID)
        args_types = [arg.type for arg in args]
        if args_types != fn_args:
            print_error(CallWithInvalidArgumentsError(
                node, fn_args, args_types))
            return TypeCheckResult(JSPDLType.INVALID)
        for index, arg in enumerate(args):
            cg.c3d_queue.append(f"param {arg.value}")
            cg.quartet_queue.append(
                Quartet(
                    Operation.PARAM,
                    Operand(value=arg.value, offset=arg.offset,
                            scope=arg.scope, op_type=arg.type),
                    Operand(scope=cg.OperandScope.LOCAL if (
                            current_fn is not None) else cg.OperandScope.GLOBAL),
                    op_options={
                        "access_register_size": st.current_symbol_table.access_register_size if st.current_symbol_table != st.global_symbol_table else 0,  # del llamador
                        "param_number": index
                    }
                )
            )
    access_register_size = 0 if st.current_symbol_table == st.global_symbol_table else st.current_symbol_table.access_register_size
    cg.c3d_queue.append(f"{cg.get_new_temporal_per_st()} := call {identifier}")
    cg.quartet_queue.append(
        Quartet(
            Operation.CALL,
            op_options={
                "tag_identifier": identifier,
                "access_register_size": access_register_size,
                "ret_type": fn.return_type,
                "tag_count": cg.functions_tag_counters[identifier]
            }
        )
    )
    cg.functions_tag_counters[identifier] += 1
    op_offset = st.current_symbol_table.access_register_size if st.current_symbol_table != st.global_symbol_table else 0
    return TypeCheckResult(
        fn.return_type,
        offset=op_offset,
        c3d_rep=cg.get_last_temporal_st(),
        scope=cg.OperandScope.LOCAL
    )


def argument_list(node: Node) -> List[TypeCheckResult] | None:
    arg_list: list[TypeCheckResult] = []
    for val in node.named_children:
        val_checked: TypeCheckResult = rule_functions[val.type](val)
        if val_checked.type == JSPDLType.INVALID:
            return None
        arg_list.append(val_checked)
    return arg_list


def do_while_statement(node: Node) -> TypeCheckResult:
    do_while_condition = node.child_by_field_name("do_while_condition")
    do_while_body = node.child_by_field_name("do_while_body")
    assert isinstance(do_while_condition, Node)
    assert isinstance(do_while_body, Node)
    tag_for_this_while = cg.while_tag_counter
    # generate tag for start of loop
    cg.quartet_queue.append(
        Quartet(Operation.WHILE_TAG, op_options={"tag": tag_for_this_while}))
    cg.while_tag_counter += 1
    # generate the condition expression
    condition_checked: TypeCheckResult = rule_functions[do_while_condition.type](
        do_while_condition)
    if (condition_checked.type == JSPDLType.INVALID
            or condition_checked.type != JSPDLType.BOOLEAN):
        return TypeCheckResult(JSPDLType.INVALID)
    cg.c3d_queue.append(f"while_start_{tag_for_this_while}:")
    body_checked = rule_functions[do_while_body.type](do_while_body)
    if body_checked.type == JSPDLType.INVALID:
        return TypeCheckResult(JSPDLType.INVALID)
    # generate the check for the condition with branch back to
    # the start of the loop if condition is true
    cg.c3d_queue.append(
        f"if {condition_checked.c3d_rep} == 1 goto while_start_{tag_for_this_while}")
    cg.quartet_queue.append(
        Quartet(
            Operation.WHILE_TRUE,
            Operand(value=condition_checked.value,
                    scope=condition_checked.scope,
                    offset=condition_checked.offset),
            op_options={"tag": tag_for_this_while},
        )
    )
    cg.while_tag_counter -= 1
    return TypeCheckResult(JSPDLType.VOID)


def if_statement(node: Node) -> TypeCheckResult:
    if_condition = node.child_by_field_name("if_condition")
    if_body = node.child_by_field_name("if_body")
    assert isinstance(if_condition, Node)
    assert isinstance(if_body, Node)
    if_condition_checked: TypeCheckResult = rule_functions[if_condition.type](
        if_condition)
    if (if_condition_checked.type != JSPDLType.BOOLEAN):
        if (if_condition_checked.type == JSPDLType.INVALID):
            return TypeCheckResult(JSPDLType.INVALID)
        print_error(TypeMismatchError(
            if_condition, JSPDLType.BOOLEAN, if_condition_checked.type))
        return TypeCheckResult(JSPDLType.INVALID)
    tag_for_this_if = cg.if_tag_counter
    # compare with 0 and jump to tag
    cg.c3d_queue.append(
        f"if {if_condition_checked.c3d_rep} == 0 goto if_false_{tag_for_this_if}"
    )
    cg.quartet_queue.append(
        Quartet(
            Operation.IF_FALSE,
            Operand(value=if_condition_checked.value,
                    scope=if_condition_checked.scope, offset=if_condition_checked.offset),
            op_options={"tag": tag_for_this_if},
        )
    )
    cg.if_tag_counter += 1
    if_body_checked: TypeCheckResult = rule_functions[if_body.type](if_body)
    if if_body_checked.type == JSPDLType.INVALID:
        return TypeCheckResult(JSPDLType.INVALID)
    cg.c3d_queue.append(f"if_false_{tag_for_this_if}:")
    # etiq
    cg.quartet_queue.append(
        Quartet(Operation.IF_TAG, op_options={"tag": tag_for_this_if}))
    return TypeCheckResult(JSPDLType.VOID)


# Get the current module
current_module = inspect.getmodule(lambda: None)

# Retrieve all functions in the current module
rule_functions = {name: func for name, func in inspect.getmembers(
    current_module, inspect.isfunction)}
