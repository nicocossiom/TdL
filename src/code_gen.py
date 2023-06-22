from enum import Enum
from typing import Any, Callable, Optional

import symbol_table as st
from symbol_table import JSPDLType

static_memory_size = 0

global_temporal_counter = -1
fn_temporal_counter = -1

is_in_global_scope = True
if_tag_counter = 0
while_tag_counter = 0
expression_tag_counter = 0
functions_tag_counters: dict[str, int] = {}

assembly = ""


def get_new_temporal_per_st() -> str:
    global global_temporal_counter, fn_temporal_counter
    if st.current_symbol_table != st.global_symbol_table:
        fn_temporal_counter += 1
        return f"t{fn_temporal_counter}"
    global_temporal_counter += 1
    return f"t{global_temporal_counter}"


def get_last_temporal_st() -> str:
    global global_temporal_counter, fn_temporal_counter
    if st.current_symbol_table != st.global_symbol_table:
        return f"t{fn_temporal_counter}"
    return f"t{global_temporal_counter}"


OpValActualValues = int | str | bool


class OpValRepType(Enum):
    """
    Possible types of representations of an OpValue
    """
    ACCUMULATOR = 1
    LITERAL = 2
    REGISTER = 3
    TEMPORAL = 4

    def __repr__(self) -> str:
        """
        Returns ACCUMULATOR, LITERAL or REGISTER in string format
        """
        return self.name.replace("OpValRep.OpValRepType.", "")


class OpValRep():
    """
    Represents how can OpValue in an Operand of a Quartet must be represented
    """

    def __init__(self, rep_type: OpValRepType, rep_value: Optional[str] = None, rep_real_size: Optional[int] = None) -> None:
        # value of the representation (e.g. "t1", ".R2", ".A")
        self.rep_value = rep_value
        # type of the representation (e.g. ACCUMULATOR, LITERAL, REGISTER)
        if rep_type == OpValRepType.ACCUMULATOR:
            self.rep_value = ".A"
        self.rep_type = rep_type
        self.rep_real_size = rep_real_size

    def __repr__(self) -> str:
        """
        Returns a representation for the operand value with type and value (e.g. "ACCUMULATOR .A", "REGISTER .R2", "TEMPORAL t1")
        """
        return f"{self.rep_type} {self.rep_value}"

    def __str__(self) -> str:
        """
        Returns a representation for the operand value with value only (e.g. ".A", ".R2", "t1")
        This is the method that python uses for representation when printing or if used in an fstring
        """
        assert self.rep_value is not None
        return self.rep_value


class OpVal:
    """
    Represents a value of an operand in a Quartet
    """

    def __init__(self, value: Optional[OpValActualValues] = None, rep: Optional[OpValRep] = None) -> None:
        self.value = value
        self.rep = rep

    def __repr__(self) -> str:
        return f"{self.value if self.value is not None else ''}{self.rep if self.rep is not None else ''}"


class Operation(Enum):
    ADD = 1
    OR = 2
    EQUALS = 3
    ASSIGN = 4
    GOTO = 5
    PARAM = 6
    CALL = 7
    RETURN = 8
    INC = 9
    PRINT = 10
    INPUT = 11
    IF_FALSE = 12
    IF_TAG = 13
    WHILE_TRUE = 14
    WHILE_TAG = 15
    FUNCTION_DECLARATION_START_TAG = 16
    FUNCTION_DECLARATION_END_TAG = 17

    def __repr__(self) -> str:
        return self.name.replace("Operation.", "")


class OperandScope(Enum):
    LOCAL = 1
    GLOBAL = 2
    TEMPORAL = 3

    def __str__(self) -> str:
        return self.name.replace("OperandScope.", "")

    def __repr__(self) -> str:
        return str(self)


class Operand:
    def __init__(self,
                 value: Optional[OpVal] = None,
                 offset: Optional[int] = None,
                 scope: OperandScope | None = None,
                 op_type: JSPDLType | None = None
                 ):
        self.value = value
        self.offset = offset
        self.scope = scope
        self.op_type = op_type

    def __repr__(self) -> str:
        return f"(value={self.value}, offset={self.offset}, scope={self.scope}, op_type={self.op_type})"


class Quartet:
    def __init__(self,
                 op: Operation,
                 op1: Operand | None = None,
                 op2: Operand | None = None,
                 res: Operand | None = None,
                 op_options: dict[str, Any] | None = None
                 ) -> None:
        self.op: Operation = op
        self.op1: Optional[Operand] = op1
        self.op2: Optional[Operand] = op2
        self.res: Optional[Operand] = res
        self.op_options = op_options

    def __repr__(self) -> str:
        props: list[str] = []
        for elem_name, elem_value in self.__dict__.items():
            if elem_value is not None:
                props.append(f"{elem_name}={elem_value}")
        elems = ",\n\t\t".join(props)
        rep = "Quartet(\n\t\t"
        return f"{rep}{elems}\n\t\t)"


# queue of quartets to be processed after code is type-checked -> all quartets are generated
quartet_queue: list[Quartet] = []
c3d_queue: list[str] = []
# list of [et DATA "someliteralstring" ... ] to be generated before
literals_queue: list[str] = []
function_queue: list[Quartet] = []
RA_stack: list[int] = []


def c3d_write_all():
    import globals
    file_path = globals.output_file.replace(".ens", ".3ic")
    with open(file_path, "w") as f:
        for code in c3d_queue:
            print(code)
            f.write(code + "\n")


class OperandPlace(Enum):
    LEFT = 1
    RIGHT = 2


def gen_operation(operation: str, q_op_1: Operand | None = None, q_op_2: Operand | None = None, op1: str | None = None, op2: str | None = None, comment: str | None = None) -> str:
    op1_p, op2_p = "", ""
    rep_1_set, rep_2_set = "", ""
    if op1 is None and q_op_1 is not None:
        rep_1_set, op1_p = find_op(q_op_1, OperandPlace.LEFT, operation)
    elif op1 is not None:
        rep_1_set, op1_p = "", op1
    if not op2 and q_op_2 is not None:
        assert q_op_2 is not None
        rep_2_set, op2_p = find_op(q_op_2, OperandPlace.RIGHT, operation)
    elif op2 is not None:
        rep_2_set, op2_p = "", op2

    sep = "," if op1_p != "" and op2_p != "" else ""
    return format_instructions(f"""
{rep_1_set}
{rep_2_set}
{operation} {op1_p} {sep} {op2_p} ; {comment}
""")


def find_op(o: Operand, place: OperandPlace, operation: str | None) -> tuple[str, str]:
    if o.scope == OperandScope.TEMPORAL:
        val = str(o.value)
        assert operation is not None
        if "lit" in val and operation != "MOVE":
            return "", val.replace("#", "/")
        return "", val
    if o.value is None:
        assert o.offset is not None
        assert o.scope in [
            OperandScope.LOCAL, OperandScope.GLOBAL], "scope invalido no es ni global ni local"
        pointer = ".IY" if o.scope == OperandScope.GLOBAL else ".IX"
        if o.offset >= 127:
            op_rep = ".R3" if place == OperandPlace.LEFT else ".R4"
            set_op = (f"""
ADD #{o.offset}, {pointer} ; {op_rep} = {pointer} + {o.offset} = operand address
MOVE .A, {op_rep}""")
            return set_op, f"[{op_rep}]"
        else:
            return "", f"#{o.offset}[{pointer}]"

    return "", str(o.value)


def gen_code():
    # static memory = global variables = local variables of main function
    global assembly
    static_memory_size = 0
    for entry in st.global_symbol_table.entries.values():
        if isinstance(entry, st.VarEntry):
            static_memory_size += entry.size

    print("3Instruction code generation:")
    print("------------------------------")
    c3d_write_all()
    print("------------------------------")
    print(f"Static memory needed to be allocated = {static_memory_size}")
    assembly += """
;-----------------------------------------------------------------------------------------
                        ORG 0
                        MOVE .SP, .IX ; initialize IX to point to the start of the stack
"""
    if static_memory_size > 0:
        assembly += format_instructions(
            """
MOVE #static_memory_start, .IY ; intialize IY to point to the start of the static memory
""")
    print("Quartets generated:")
    for q in quartet_queue:
        # call the function that generates the code for the operation
        print(f"\t{q}")
        assembly += code_gen_dict[q.op](q)
    assembly += """
                        HALT
                        NOP
"""
    for fn in function_queue:
        assembly += code_gen_dict[fn.op](fn)
    assembly += "\n".join(literals_queue)  # add literals to assembly
    if static_memory_size > 0:
        assembly += "static_memory_start:\n"
        assembly += format_instructions(
            f"""
RES {static_memory_size}{" " }; reserve {static_memory_size} memory addresses for global variables
        """)
    assembly += gen_instr("END", "end of program")
    assembly += """                
;----------------------------------------------------------------------------------------
    """
    print(f"Assembly code:{assembly}")
    import globals
    print("Assembly code generated successfully, written to .ens file")
    with open(globals.output_file, "w") as f:
        f.write(assembly)


def gen_instr_and_add_to_assembly(ins: str, comment: str = "") -> None:
    global assembly
    assembly += gen_instr(ins, comment)


# the column where the comment starts
comment_column_start = 69


def gen_instr(ins: str, comment: str = "") -> str:
    padding = " " * 24
    comment = f"; {comment}" if comment else ""
    comment_padding = " " * (comment_column_start - len(ins) - len(padding))
    return f"{padding}{ins}{comment_padding}{comment}\n"


def format_instructions(ins: str) -> str:
    padding = " " * 24
    # separate the ins per new line into a list
    instrs = ins.split("\n")
    instrs_formatted = ""
    for instr in instrs:
        if instr.rstrip() == "":
            continue
        if ":" in instr:
            # its a label
            instrs_formatted += f"{instr}\n"
            continue
        instr_split = instr.split(";")
        comment_padding = " " * \
            (comment_column_start - len(instr_split[0]) - len(padding))
        comment = f"; {instr_split[1]}" if len(
            instr_split) == 2 else ""
        instrs_formatted += f"{padding}{instr_split[0]}{comment_padding}{comment}\n"
    return instrs_formatted


def gen_add(q: Quartet) -> str:
    if not q.op1 or not q.op2 or not q.res:
        raise CodeGenException(
            "Addition operation must have at least two operands and a result")
    instrs = "; empieza +\n"
    instrs += gen_operation("ADD", q.op1, q.op2, comment=" add op1, op2")
    instrs += gen_operation("MOVE", q_op_2=q.res, op1=".A",
                            comment="move accumulator to result")
    instrs += "; termina +\n"
    return instrs


def gen_inc(q: Quartet) -> str:
    if not q.op1:
        raise CodeGenException(
            "Increment operation must have an operand")
    return gen_operation("INC", q.op1, comment=" op1 ++")


def gen_or(q: Quartet) -> str:
    if not q.op1 or not q.op2 or not q.res:
        raise CodeGenException(
            "OR operation must have at least two operands and a result")
    instrs = "; empieza || \n"
    instrs += gen_operation("OR", q.op1, q.op2, comment=" or op1, op2")
    instrs += gen_operation("MOVE", q_op_2=q.res, op1=".A",
                            comment="move accumulator to result")
    instrs += "; termina ||\n"
    return instrs


def gen_equals(q: Quartet) -> str:
    if not q.op1 or not q.op2 or not q.res:
        raise CodeGenException(
            "Equals operation must have at least two operands and a result")
    instrs = "; empieza ==\n"
    instrs += gen_operation("CMP", q.op1, q.op2, comment=" compare op1, op2")
    instrs += gen_operation("BZ ", op1="$5",
                            comment="true #1 (5 bc opcode1+op1.1+op1.2+opcode2+op2.1 ")
    instrs += gen_operation("MOVE", op1="#0",
                            q_op_2=q.res, comment="equal ? false")
    instrs += gen_operation("BR ", op1="$3",
                            comment="skip next instr (3 bc opcode+op1+op2)")
    instrs += gen_operation("MOVE", op1="#1",
                            q_op_2=q.res, comment="equal ? true")
    instrs += "; termina ==\n"
    return instrs


def gen_assign(q: Quartet) -> str:
    # op1 es la expresion que se asigna
    # res es la variable a la que se asigna
    if not q.op1 or not q.res:
        raise CodeGenException(
            "Assign operation must have at least one operand and a result")
    if q.res.op_type == JSPDLType.STRING:
        assert q.res.offset is not None
        code = ";empieza asignacion de string\n"
        pointer_res = get_pointer_from_operand_scope(q.res)
        if q.op1.scope == OperandScope.TEMPORAL:
            assert q.op1.value is not None
            assert q.op1.value.rep is not None
            assert q.op1.value.rep.rep_real_size is not None
            # right expression is a constant
            code += format_instructions(f"""
MOVE {q.op1.value}, .R5; poner en .R5 la direccion de LECTURA
ADD #{q.res.offset}, {pointer_res}
MOVE .A, .R6; poner en R6 la direccion de ESCRITURA
{copy_operand(q.op1.value.rep.rep_real_size)}""")
        else:
            # right expression is a variable
            assert q.res.offset is not None
            assert q.op1.offset is not None
            pointer_op1 = get_pointer_from_operand_scope(q.op1)
            pointer_res = get_pointer_from_operand_scope(q.res)
            code += format_instructions(f"""
ADD #{q.op1.offset}, {pointer_op1}
MOVE .A, .R5; poner en R5 la direccion de LECTURA
ADD #{q.res.offset}, {pointer_res}
MOVE .A, .R6; poner en R6 la direccion de ESCRITURA
{copy_operand()}""")
        code += ";termina asignacion de string\n"
        return code
    return gen_operation("MOVE", q.op1, q.res, comment=" assign op1 to res")


def gen_goto(q: Quartet) -> str:
    if not q.op1:
        raise CodeGenException(
            "Goto operation must recieve an operand")

    return gen_instr(f"BR ${q.op1.value}", "jumps to the tag specified in the operand")


if_error_msg = "If operation must have at least one operand and a result"


def gen_if_false_goto(q: Quartet) -> str:
    if q.op_options is None:
        raise CodeGenException(if_error_msg)
    if q.op_options.get("tag") is None:
        raise CodeGenException(if_error_msg)
    if not q.op1:
        raise CodeGenException(
            "If_False_Goto operation must recieve an operand")
    given_tag_counter: int = q.op_options["tag"]
    instrs = "; empieza if\n"
    instrs += gen_operation("CMP", q.op1, op2="#0",
                            comment="compare if condition, if cmp 0 means false")
    instrs += gen_operation("BZ", op1=f"$if_tag{given_tag_counter}",
                            comment="if comparision is false, jump after the if block")
    instrs += "; termina if\n"
    return instrs


def gen_if_tag(q: Quartet) -> str:
    if q.op_options is None:
        raise CodeGenException(if_error_msg)
    if q.op_options.get("tag") is None:
        raise CodeGenException(if_error_msg)
    given_tag_counter: int = q.op_options["tag"]
    ret_val = f"if_tag{given_tag_counter}:\n"
    ret_val += gen_instr("NOP", "NOP to avoid two if consecutive labels")
    return ret_val


def gen_while_true_goto(q: Quartet) -> str:
    if q.op_options is None:
        raise CodeGenException(while_error_message)
    if q.op_options.get("tag") is None:
        raise CodeGenException(while_error_message)
    if not q.op1:
        raise CodeGenException(
            "While_True_Goto operation must recieve an operand")
    given_tag_counter: int = q.op_options["tag"]
    instrs = "; empieza while\n"
    instrs += gen_operation("CMP", q.op1, op2="#1",
                            comment="compare while condition, if cmp 1 means true")
    instrs += gen_operation("BZ", op1=f"$while_tag{given_tag_counter}",
                            comment="if while condition is true, jump at the beginning of the block")
    instrs += "; termina while\n"
    return instrs


while_error_message = "While operation must have at least one operand and a result"


def gen_while_tag(q: Quartet) -> str:
    if q.op_options is None:
        raise CodeGenException(while_error_message)
    if q.op_options.get("tag") is None:
        raise CodeGenException(while_error_message)
    given_tag_counter: int = q.op_options["tag"]

    ret_val = f"while_tag{given_tag_counter}:\n"
    ret_val += gen_instr("NOP", "NOP to avoid two while consecutive labels")

    return ret_val


def gen_input(q: Quartet) -> str:
    if not q.op1:
        raise CodeGenException(
            "Input operation must have at least one operand")

    if q.op1.op_type == JSPDLType.INT:  # is an integer or boolean
        return gen_operation("ININT", q.op1, comment="input integer")
    return gen_operation("INSTR", q.op1, comment="input string")


def gen_print(q: Quartet) -> str:
    if not q.op1:
        raise CodeGenException(
            "Print operation must have at least one operand")

    if q.op1.op_type == JSPDLType.STRING:  # is string
        return gen_operation("WRSTR", q.op1, comment=" print string")
    return gen_operation("WRINT", q.op1, comment=" print int")


def gen_function_start_tag(q: Quartet) -> str:
    if not q.op_options:
        raise CodeGenException(
            "Function_tag operation must have op_options with defined tag_identifier")
    identifier = q.op_options["tag_identifier"]

    return format_instructions(f"""
BR ${identifier}_end ; jump to the end of the function to avoid calling it 
{identifier}:
NOP; NOP to manage recursive functions
""")


def gen_function_end_tag(q: Quartet) -> str:
    if not q.op_options:
        raise CodeGenException(
            "Function_end_tag operation must have op_options with defined tag_identifier")
    identifier = q.op_options["tag_identifier"]

    return format_instructions(f"""
{identifier}_end:
NOP ; para evitar 2 tags juntos
""")


def get_pointer_from_operand_scope(op: Operand) -> str:
    if op.scope == OperandScope.GLOBAL:
        pointer = ".IY"
    else:
        pointer = ".IX"
    return pointer


copy_tag = -1


def copy_operand(size: int | None = None) -> str:
    if size is None:
        size = 62
    global copy_tag
    copy_tag += 1
    instr = f""" 
MOVE #0, .R7; r7 = contador para salir del bucle de asignacion de strings

; copiar los datos de op1 a res
copy_{copy_tag}_loop:
MOVE [.R5], [.R6] ; #offset[/op1] -> #offset[/res]
INC .R5; aumentamos la posicion de la que leemos
INC .R6;  aumentamos la posicion en la que escribimos
INC .R7; aumentamos el contador del bucle
CMP #{size} , .R7
BP /copy_{copy_tag}_loop; si R7 <63 repites el bucle
"""
    return instr


def gen_function_param(q: Quartet) -> str:
    if not q.op1 or not q.op2:
        raise CodeGenException(
            "Parameter operation must have at least one operand")
    if not q.op_options:
        raise CodeGenException(
            "Parameter operation must have op_options with defined AC size")
    access_register_size: int = q.op_options["access_register_size"]
    param_number: int = q.op_options["param_number"]
    arg_offset: int = q.op_options["arg_offset"]
    assert q.op1.op_type is not None
    pointer = get_pointer_from_operand_scope(q.op2)
    instrs = f"; empieza paso de parámetro {param_number}\n"
    copy_val: int | None = None
    if q.op1.op_type != JSPDLType.STRING:
        param_str = gen_operation(
            "MOVE", q.op1, op2="[.A]", comment=" Pasamos el param a su posicion, la suma anterior incrementa por cada param")
    else:

        if q.op1.value is not None and q.op1.value.rep is not None and q.op1.value.rep.rep_value is not None:
            param = f"MOVE {q.op1.value.rep.rep_value}, .R5 ; R5 = direccion param {param_number} literal = direccion de LECTURA"
            copy_val = q.op1.value.rep.rep_real_size
        else:
            param = f"""
ADD #{q.op1.offset}, {pointer}
MOVE .A, .R5; R5 = direccion param {param_number} referencia = direccion de LECTURA"""

        param_str = format_instructions(f"""
MOVE .A, .R6; R6 = direccion de ESCRITURA = direccion del param {param_number} de la funcion llamada
{param}
{copy_operand(copy_val)}""")

    instrs += format_instructions(f"""
ADD #{access_register_size}, .IX ; movemos el puntero al siguiente RA
ADD #{arg_offset+1}, .A; Nos colocamos en la parte de params del RA del llamado""")
    instrs += param_str
    instrs += f";fin de paso de parametro {param_number}\n"
    return instrs


def gen_function_return(q: Quartet) -> str:
    ret_val = ""
    salto = gen_operation(
        "BR", op1="[.IX]", comment="devuelve el control al llamador")

    if not q.op1:
        return salto
    if (q.op1.scope == OperandScope.GLOBAL and q.op1.offset is None):
        return gen_instr("HALT", "si en el programa principal haces un return, paras de ejecutar")
    if not q.op_options:
        raise CodeGenException(
            "Function_return operation must have op_options with defined AC size")

    # local return from function
    access_register_size: int = q.op_options["access_register_size"]
    assert q.op1.op_type is not None
    if q.op1.op_type == JSPDLType.STRING:
        if q.op1.value is not None and q.op1.value.rep is not None and q.op1.value.rep.rep_type.LITERAL:
            ret_val += gen_operation("MOVE", q.op1, op2=".R5",
                                     comment=" R5 = direccion de LECTURA = direccion del valor de retorno")
        elif q.op1.scope == OperandScope.GLOBAL:
            ret_val += format_instructions(f"""
ADD #{q.op1.offset}, .IY ; IY = direccion del valor de retorno
MOVE .A, .R5; R5 = direccion de LECTURA = direccion del valor de retorno
""")

    else:
        ret_val += ";conmienzo return\n"
        if q.op1.scope == OperandScope.GLOBAL:
            ret_val += format_instructions(f"""
ADD #{q.op1.offset}, .IY ; IY = direccion del valor de retorno
MOVE .A, .R5; R5 = direccion de LECTURA = direccion del valor de retorno
""")
        else:
            ret_val += format_instructions(f""" 
SUB #{access_register_size}, #{st.size_dict[q.op1.op_type]}  ; tamaño del valor devuelto = {st.size_dict[q.op1.op_type]}
ADD .A, .IX ; A contiene la dirección del valor de retorno
MOVE .A, .R5; R5 = direccion de LECTURA = direccion del valor de retorno""")
        ret_val += gen_operation("MOVE", q.op1,
                                 op2="[.A]", comment="Colocar valor de retorno")
    ret_val += salto
    ret_val += ";fin return\n"
    return ret_val


def gen_function_call(q: Quartet) -> str:
    if not q.op_options:
        raise CodeGenException(
            "Function_tag operation must have op_options with defined tag_identifier")
    identifier = q.op_options["tag_identifier"]
    access_register_size: int = q.op_options["access_register_size"]
    tag_count: int = q.op_options["tag_count"]
    try:
        function_tag = f"{identifier}"
        ret_tag = f"ret_dir_{identifier}_{tag_count}"
    except KeyError:
        raise CodeGenException(
            "Function_tag operation must have op_options with defined tag_identifier which must correspond to a function_tag")
    instr = f";empieza llamda a {identifier}() \n"
    instr += format_instructions(f"""
ADD #{access_register_size}, .IX ; avanzo el puntero de pila al RA de la funcion llamada   
MOVE .A, .IX; recoloco el puntero de pila al comienzo del resgistro de activacion al llamado

MOVE #{ret_tag}, [.IX]; coloco la dirección del salto de retorno en el EM del RA de la funcion llamada 
BR /{function_tag}; salto al codigo de la funcion llamada
""")

    ret_val = ""
    if q.op_options["ret_type"] != JSPDLType.VOID:
        # hay valor de retorno
        ret_type = q.op_options["ret_type"]
        if ret_type == JSPDLType.STRING:
            ret_val = (f"""
ADD #{access_register_size}, .IX
; nosotros en .R6 tenemos que dejar la direccion donde copiarlo, esto nuestro RA : EM + param + locales 
MOVE .A, .R6; poner en R6 la direccion de ESCRITURA
; llamado deja en .R5 la direccion del valor de retorno = direccion de LECTURA
{copy_operand()}
""")
        else:
            ret_val = (f"""
; llamado deja en .R5 la direccion del valor de retorno
; llamamos a copia y le decimos que nos lo deje en temporal
MOVE [.R5], .R8 ; R8 = valor de retorno

; ponemos el valor de retorno en la temporal
MOVE .R8, #{access_register_size}[.IX]
""")

    instr += format_instructions(f"""
{ret_tag}: 
SUB .IX, #{access_register_size} 
MOVE .A, .IX ; recolocamos el puntero de pila en el EM del llamado
{ret_val}
    """)
    instr += f";fin llamada a {identifier}() \n"
    return instr


code_gen_dict: dict[Operation, Callable[[Quartet], str]] = {
    Operation.ADD: gen_add,
    Operation.OR: gen_or,
    Operation.EQUALS:  gen_equals,
    Operation.ASSIGN:  gen_assign,
    Operation.GOTO:  gen_goto,
    Operation.PARAM:  gen_function_param,
    Operation.RETURN:  gen_function_return,
    Operation.CALL:  gen_function_call,
    Operation.INC:  gen_inc,
    Operation.PRINT:  gen_print,
    Operation.INPUT:  gen_input,
    Operation.IF_FALSE:  gen_if_false_goto,
    Operation.IF_TAG:  gen_if_tag,
    Operation.WHILE_TRUE:  gen_while_true_goto,
    Operation.WHILE_TAG:  gen_while_tag,
    Operation.FUNCTION_DECLARATION_START_TAG:  gen_function_start_tag,
    Operation.FUNCTION_DECLARATION_END_TAG:  gen_function_end_tag,
}


class CodeGenException(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):

        return self.msg
