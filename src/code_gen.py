from enum import Enum
from typing import Callable, Optional

import symbol_table as st
from symbol_table import JSPDLType

static_memory_size = 0
static_memory_current_offset = 0

temporal_counter = 0

is_in_global_scope = True


def get_new_temporal() -> str:
    global temporal_counter
    temporal_counter += 1
    return f"t{temporal_counter}"


def get_last_temporal() -> str:
    global temporal_counter
    return f"t{temporal_counter}"


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
                 res: Operand | None = None
                 ) -> None:
        self.op: Operation = op
        self.op1: Optional[Operand] = op1
        self.op2: Optional[Operand] = op2
        self.res: Optional[Operand] = res

    def __repr__(self) -> str:
        return f"({self.op}\n\t\top1={self.op1},\n\t\top2={self.op2}\n\t\tres={self.res}\n\t)"


# queue of quartets to be processed after code is type-checked -> all quartets are generated
quartet_queue: list[Quartet] = []
c3d_file = open("out.3ic", "w")
c3d_queue: list[str] = []
# list of [et DATA "someliteralstring" ... ] to be generated before
literals_queue: list[str] = []

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
    static_memory_size = st.global_symbol_table.current_offset

    print("3Instruction code generation:")
    print("------------------------------")
    c3d_write_all()
    print("------------------------------")
    print(f"Static memory needed to be allocated = {static_memory_size}")
    assembly = """
;-----------------------------------------------------------------------------------------
                        ORG 0
"""
    if static_memory_size > 0:
        assembly += gen_instr("MOVE #static_memory_start, .IY",
                              "intialize IY to point to the start of the static memory")

    print("Quartets generated:")
    for q in quartet_queue:
        # call the function that generates the code for the operation
        print(f"\t{q}")
        assembly += code_gen_dict[q.op](q)
    assembly += """
                        HALT
"""
    assembly += "\n".join(literals_queue)  # add literals to assembly
    if static_memory_size > 0:
        assembly += f"""
        static_memory_start:    RES {static_memory_size} ; reserve {static_memory_size} memory addresses for global variables
        """
    assembly += """
                        END
;----------------------------------------------------------------------------------------
    """
    print(f"Assembly code:{assembly}")

    print("Assembly code generated successfully, written to .ens file")
    with open("out.ens", "w") as f:
        f.write(assembly)


def gen_instr(ins: str, comment: str = "") -> str:
    padding = " " * 24
    return f"{padding}{ins}\t\t\t;{comment}\n"


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
    global static_memory_current_offset
    if not q.op1 or not q.res:
        raise CodeGenException(
            "Assign operation must have at least one operand and a result")

    return gen_instr(f"MOVE {find_op(q.res)},{find_op(q.op1)}", "ASSIGN op1, res")


def gen_goto(q: Quartet) -> str:
    if not q.op1:
        raise CodeGenException(
            "Goto operation must recieve an operand")

    return gen_instr(f"BR ${q.op1.value}", "jumps to the tag specified in the operand")


def gen_param(q: Quartet) -> str:
    if not q.op1:
        raise CodeGenException(
            "Parameter operation must have at least one operand")
    else:
        return \
            gen_instr(f"ADD , .IX", "ADD #Tam_RA_llamador, .IX") + \
            gen_instr(f"ADD #1, .A", "ADD #1, .A") + \
            gen_instr(f"MOVE {find_op(q.op1)} [.A], ", "MOVE op1, [.A];")


def gen_return(q: Quartet) -> str:
    return ""


def gen_call(q: Quartet) -> str:
    return ""


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


code_gen_dict: dict[Operation, Callable[[Quartet], str]] = {
    Operation.ADD: gen_add,
    Operation.OR: gen_or,
    Operation.EQUALS:  gen_equals,
    Operation.ASSIGN:  gen_assign,
    Operation.GOTO:  gen_goto,
    Operation.PARAM:  gen_param,
    Operation.RETURN:  gen_return,
    Operation.CALL:  gen_call,
    Operation.INC:  gen_inc,
    Operation.PRINT:  gen_print,
    Operation.INPUT:  gen_input,
}


class CodeGenException(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):

        return self.msg
