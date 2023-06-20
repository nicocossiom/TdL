import codecs
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

    def __repr__(self) -> str:
        """
        Returns ACCUMULATOR, LITERAL or REGISTER in string format
        """
        return self.name.replace("OpValRep.OpValRepType.", "")


class OpValRep():
    """
    Represents how can OpValue in an Operand of a Quartet must be represented
    """

    def __init__(self, rep_type: OpValRepType, rep_value: Optional[str] = None) -> None:
        # value of the representation (e.g. "t1", ".R2", ".A")
        self.rep_value = rep_value
        # type of the representation (e.g. ACCUMULATOR, LITERAL, REGISTER)
        if rep_type == OpValRepType.ACCUMULATOR:
            self.rep_value = ".A"
        self.rep_type = rep_type

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
c3d_file = open("out.3ic", "w")
c3d_queue: list[str] = []
# list of [et DATA "someliteralstring" ... ] to be generated before
literals_queue: list[str] = []
function_queue: list[Quartet] = []
RA_stack: list[int] = []


def c3d_write_all():
    for code in c3d_queue:
        print(code)
        c3d_file.write(code + "\n")


def find_op(o: Operand) -> str:
    if o.scope == OperandScope.TEMPORAL:
        return str(o.value)  # return the representation given by OpValRep
    if o.value is None:
        assert o.offset is not None
        if o.scope == OperandScope.GLOBAL:
            return f"#{o.offset}[.IY]"
        elif o.scope == OperandScope.LOCAL:
            return f"#{o.offset}[.IX]"
        else:
            raise CodeGenException(
                f"Operand {o} has an invalid scope {o.scope}")
    return f"#{o.value}"


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
        assembly += gen_instrs(
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
"""
    for fn in function_queue:
        assembly += code_gen_dict[fn.op](fn)
    assembly += "\n".join(literals_queue)  # add literals to assembly
    if static_memory_size > 0:
        assembly += "static_memory_start:"
        assembly += gen_instrs(
            f"""
RES {static_memory_size}{" " }; reserve {static_memory_size} memory addresses for global variables
        """)
    assembly += gen_instr("END", "end of program")
    assembly += """                
;----------------------------------------------------------------------------------------
    """
    print(f"Assembly code:{assembly}")

    print("Assembly code generated successfully, written to .ens file")
    with open("out.ens", "w") as f:
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


def gen_instrs(ins: str) -> str:
    padding = " " * 24
    # separate the ins per new line into a list
    instrs = ins.split("\n")
    instrs_formatted = ""
    for instr in instrs:
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

    return gen_instr(f"ADD {find_op(q.op1)}, {find_op(q.op2)}", "ADD op1, op2")


def gen_inc(q: Quartet) -> str:
    if not q.op1:
        raise CodeGenException(
            "Increment operation must have an operand")

    return gen_instr(f"INC {find_op(q.op1)}", "INC op1")


def gen_or(q: Quartet) -> str:
    if not q.op1 or not q.op2 or not q.res:
        raise CodeGenException(
            "OR operation must have at least two operands and a result")
    else:
        return gen_instr(f"OR {find_op(q.op1)}, {find_op(q.op2)}", "OR op1, op2")


def gen_equals(q: Quartet) -> str:
    if not q.op1 or not q.op2 or not q.res:
        raise CodeGenException(
            "Equals operation must have at least two operands and a result")
    else:
        return \
            gen_instr(f"CMP {find_op(q.op1)}, {find_op(q.op2)}", "CMP op1, op2") + \
            gen_instr("BZ $5",  "true #1. (5 bc opcode1+op1.1+op1.2+opcode2+op2.1 ) ") + \
            gen_instr(f"MOVE #0, {find_op(q.res)}", " equal ? false") + \
            gen_instr("BR $3", "skip next instr (3 bc opcode+op1+op2)") + \
            gen_instr(f"MOVE #1, {find_op(q.res)}", "equal ? true")


def gen_assign(q: Quartet) -> str:
    global assembly
    # op1 es la variable es a la que se le asigna
    # res es el valor que se le asigna, que viene de otra variable o de una constante
    if not q.op1 or not q.res:
        raise CodeGenException(
            "Assign operation must have at least one operand and a result")
    if q.op1.op_type == JSPDLType.STRING:
        # convert the string to a list of ascii codes and move them to memory
        if q.op_options is None:
            raise CodeGenException(
                "When assigning a string the access register size must be specified")
        assert q.op1.offset is not None
        assembly = ""
        left_ar_pointer = ".IX" if q.op1.scope == OperandScope.LOCAL else ".IY"
        if q.res.scope == OperandScope.TEMPORAL:
            # right expression is a constant
            assert q.res.value is not None
            assert q.res.value.rep is not None
            assert q.res.value.rep.rep_value is not None
            assert isinstance(q.res.value.rep.rep_value, str)
            ascii_codes = codecs.escape_decode(
                q.res.value.rep.rep_value)[0]
            for byte_counter, (byte, char) in enumerate(zip(ascii_codes, q.res.value.rep.rep_value)):
                char = '\\n' if char == '\n' else char
                char = '\\t' if char == '\t' else char
                assembly += gen_instr(
                    f"MOVE #{byte}, #{q.op1.offset + byte_counter}[{left_ar_pointer}]", f"assigning char '{char}'")
            assembly += gen_instr(
                f"MOVE #0, #{q.op1.offset + len(ascii_codes)}[{left_ar_pointer}]", "null terminator")
        else:
            right_ar_pointer = ".IX" if q.res.scope == OperandScope.LOCAL else ".IY"
            # right expression is a variable
            assert q.res.offset is not None
            assert q.op1.offset is not None
            pointer = get_pointer_from_operand_scope(q.op1)
            pointer = get_pointer_from_operand_scope(q.res)
            assembly += f"""
ADD #{q.op1.offset}, {pointer}
MOVE .A, .R6; poner en R6 la direccion de ESCRITURA
ADD #{q.res.offset}, {pointer}
MOVE .A, .R5; poner en R5 la direccion de LECTURA

        {copy_operand()}
        """

        return assembly
    return gen_instr(f"MOVE {find_op(q.res)},{find_op(q.op1)}", "ASSIGN op1, res")


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
    return gen_instr(f"CMP {find_op(q.op1)}, #0", "compare if condition, if cmp 0 means false") + \
        gen_instr(f"BZ $if_tag{given_tag_counter}",
                  "if comparision is false, jump after the if block")


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
    ret_val = gen_instr(f"CMP {find_op(q.op1)}, #1", "compare while condition, if cmp 1 means true") + \
        gen_instr(f"BZ $while_tag{given_tag_counter}",
                  "if while condition is true, jump at the beginning of the block")
    return ret_val


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
        return gen_instr(f"ININT {find_op(q.op1)}", "ININT op1")
    else:  # is a string
        return gen_instr(f"INSTR {find_op(q.op1)}", "INSTR op1")


def gen_print(q: Quartet) -> str:
    if not q.op1:
        raise CodeGenException(
            "Print operation must have at least one operand")

    if q.op1.op_type == JSPDLType.STRING:  # is string
        return gen_instr(f"WRSTR {find_op(q.op1)}", "WRSTR op1")

    else:  # boolean or int
        if q.op1.scope == OperandScope.TEMPORAL:
            return gen_instr(f"MOVE {find_op(q.op1)}, .R6", "save literal int  in R6") + \
                gen_instr("WRINT .R6", "write literal int stored .R6 to console")
        else:
            return gen_instr(f"WRINT {find_op(q.op1)}", "WRINT op1")


def gen_function_start_tag(q: Quartet) -> str:
    if not q.op_options:
        raise CodeGenException(
            "Function_tag operation must have op_options with defined tag_identifier")
    identifier = q.op_options["tag_identifier"]

    return gen_instrs(f"""
BR ${identifier}_end ; jump to the end of the function to avoid calling it 
{identifier}:
NOP; NOP to manage recursive functions
""")


def gen_function_end_tag(q: Quartet) -> str:
    if not q.op_options:
        raise CodeGenException(
            "Function_end_tag operation must have op_options with defined tag_identifier")
    identifier = q.op_options["tag_identifier"]

    return f"{identifier}_end:"


def get_pointer_from_operand_scope(op: Operand) -> str:
    if op.scope == OperandScope.GLOBAL:
        pointer = ".IY"
    else:
        pointer = ".IX"
    return pointer


copy_tag = -1


def copy_operand() -> str:
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
CMP #62 , .R7
BP /copy_{copy_tag}_loop; si R7 <63 repites el bucle
"""
    return instr
#         param_str = (f"""
# {copy_operand(q.op1, Operand(scope=OperandScope.LOCAL, offset=(param_number * st.size_dict[q.op1.op_type])+1))}

# """


def gen_function_param(q: Quartet) -> str:
    if not q.op1 or not q.op2:
        raise CodeGenException(
            "Parameter operation must have at least one operand")
    if not q.op_options:
        raise CodeGenException(
            "Parameter operation must have op_options with defined AC size")
    access_register_size: int = q.op_options["access_register_size"]
    param_number: int = q.op_options["param_number"]
    assert q.op1.op_type is not None
    pointer = get_pointer_from_operand_scope(q.op2)
    if q.op1.op_type == JSPDLType.STRING:
        if q.op1.value is not None and q.op1.value.rep is not None and q.op1.value.rep.rep_value is not None:

            lit_n = len(literals_queue)
            literals_queue.append(
                gen_instr(
                    f"lit{lit_n+1}: DATA \"{q.op1.value.rep.rep_value}\"")
            )
            param = f"""
MOVE #lit{lit_n+1}, .R5 ; poner en R5 ladireccion de LECTURA
"""
        else:
            param = f"""
MOVE .A, .R5; poner en R5 la direccion de LECTURA
ADD #{q.op1.offset}, {pointer}
"""
        param_str = f"""
MOVE .A, .R6; poner en R6 la direccion de ESCRITURA
{param}

        {copy_operand()}
        """
    else:
        param_str = f"MOVE {find_op(q.op1)},[.A] ; Pasamos el param a su posicion, la suma anterior incrementa por cada param"
    return gen_instrs(f"""
ADD #{access_register_size}, .IX ; movemos el puntero al siguiente RA
ADD #{(param_number * st.size_dict[q.op1.op_type])+1}, .A; Nos colocamos en la parte de params del RA del llamado
{param_str}
""")


def gen_function_return(q: Quartet) -> str:
    ret_val = ""

    if not q.op_options:
        raise CodeGenException(
            "Function_return operation must have op_options with defined AC size")

    if q.op1:
        if (q.op1.scope == OperandScope.GLOBAL):
            return gen_instr("HALT", "si en el programa principal haces un return, paras de ejecutar")
        # local return from function
        access_register_size: int = q.op_options["access_register_size"]
        assert q.op1.op_type is not None
        ret_val += gen_instrs(f""" 
SUB #{access_register_size}, #{st.size_dict[q.op1.op_type]}  ; tamaño del valor devuelto = {st.size_dict[q.op1.op_type]}
ADD .A, .IX ; A contiene la dirección del valor de retorno
MOVE {find_op(q.op1)}, [.A]; Y es el desplazamiento de op1 en la TS
        """)
    ret_val += gen_instr("BR [.IX] ;devuelve el control al llamador")

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
    instr = gen_instrs(f"""
ADD #{access_register_size}, .IX ; avanzo el puntero de pila al RA de la funcion llamada   
MOVE .A, .IX; recoloco el puntero de pila al comienzo del resgistro de activacion al llamado

MOVE #{ret_tag}, [.IX]; coloco la dirección del salto de retorno en el EM del RA de la funcion llamada 

BR /{function_tag}; salto al codigo de la funcion llamada
""")

    if q.op_options["ret_type"] != JSPDLType.VOID:
        # hay valor de retorno
        ret_type = q.op_options["ret_type"]
    else:
        instr += gen_instrs(f"""
{ret_tag}: 
SUB .IX, #{access_register_size} 
MOVE .A, .IX ; recolocamos el puntero de pila en el EM del llamado
    """)
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
