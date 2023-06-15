from enum import Enum
from typing import Callable, Optional

static_memory_size = 0
static_memory_current_offset = 0


class Operation(Enum):
    ADD = 1
    OR = 2
    EQUALS = 3
    ASSIGN = 4
    GOTO = 5
    PARAM = 6
    CALL = 7
    RETURN = 8

    def __repr__(self) -> str:
        return self.name


class OperandScope(Enum):
    LOCAL = 1
    GLOBAL = 2
    TEMPORAL = 3

    def __str__(self) -> str:
        return self.name.replace("OperandScope.", "")

    def __repr__(self) -> str:
        return str(self)


OpVal = int | str | bool | None
OpOffset = int | None


class Operand:
    def __init__(self, value: OpVal = None, offset: OpOffset = None, scope: OperandScope | None = None):
        self.value = value
        self.offset = offset
        self.scope = scope

    def __repr__(self) -> str:
        rep = "("
        for attr in self.__dict__:
            if attr in ["value", "offset", "scope"] and self.__dict__[attr] is not None:
                rep += f"{attr}={self.__dict__[attr]},"
        rep += ")"
        return rep


class Quartet:
    def __init__(self, op: Operation, op1: Operand | None = None, op2: Operand | None = None, res: Operand | None = None) -> None:
        self.op: Operation = op
        self.op1: Optional[Operand] = op1
        self.op2: Optional[Operand] = op2
        self.res: Optional[Operand] = res

    def __repr__(self) -> str:
        return f"({self.op}, op1={self.op1}, op2={self.op2}, res={self.res})"


# queue of quartets to be processed after code is type-checked -> all quartets are generated
quartet_queue: list[Quartet] = []
c3d_file = open("out.3ic", "w")
c3d_queue: list[str] = []


def c3d_write_all():
    for code in c3d_queue:
        print(code)
        c3d_file.write(code + "\n")


# def count_global_memory():
#     for q in quartet_queue:
#         if q.op1 and q.op1.scope == OperandScope.GLOBAL:
#             global static_memory_end
#             static_memory_end += 1


def gen_code():
    # count_global_memory()
    print("3Instruction code generation:")
    print("------------------------------")
    c3d_write_all()
    print("------------------------------")
    print(f"Static memory needed to be allocated = {static_memory_size}")
    print("Assembly code:")
    assembly = """
;-----------------------------------------------------------------------------------------
                        ORG 0
                        MOVE #static_memory_start, .IY
"""
    print("Quartets generated:")
    for q in quartet_queue:
        # call the function that generates the code for the operation
        print(f"\t{q}")
        assembly += code_gen_dict[q.op](q)
    assembly += f"""
                        HALT
static_memory_start:    RES {static_memory_size} ; reserve {static_memory_size} memory addresses for global variables
                        END
;----------------------------------------------------------------------------------------
    """
    print(f"final assembly: \n{assembly}")
    print("Assembly code generated successfully, generating .ens file...")
    with open("out.ens", "w") as f:
        f.write(assembly)


def gen_instr(ins: str) -> str:
    padding = " " * 24
    return f"{padding}{ins}\n"


def gen_add(q: Quartet) -> str:
    if not q.op1 or not q.op2 or not q.res:
        raise CodeGenException(
            "Addition operation must have at least two operands and a result")

    # si esta en TS global, acceder por .IY si esta en TS local acceder por .IX
    return ""


def gen_or(q: Quartet) -> str:
    return ""


def gen_equals(q: Quartet) -> str:
    return ""


def find_op(o: Operand) -> str:
    if o.scope == OperandScope.GLOBAL:

        return ".IY"
    elif o.scope == OperandScope.LOCAL:
        return ".IX"
    else:
        raise CodeGenException(
            f"Operand {o} has an invalid scope {o.scope}")
    return ""


def gen_assign(q: Quartet) -> str:
    global static_memory_current_offset
    if not q.op1 or not q.res:
        raise CodeGenException(
            "Assign operation must have at least one operand and a result")

    if q.op1.scope == OperandScope.GLOBAL:
        if q.op1.offset is None:
            raise CodeGenException(
                "Global variable must have an offset to be accessed")
        op_processed = ""
        if q.op1.value is None:
            assert q.op1.offset is not None
            op_processed = "#" + str(
                q.op1.offset - 1) + "[.IY]" if q.op1.scope == OperandScope.GLOBAL else "#" + str(q.op1.offset-1) + "[.IX]"
        else:
            op_processed = "#"+str(q.op1.value)
        static_memory_current_offset += q.op1.offset
        print(f"\tassign: dir 0x{static_memory_current_offset} = {q.res}")
        res_processed = ""
        if q.res.value is None:
            assert q.res.offset is not None
            res_processed = "#" + str(
                q.res.offset - 1) + "[.IY]" if q.res.scope == OperandScope.GLOBAL else "#" + str(q.res.offset-1) + "[.IX]"
        else:
            res_processed = "#"+str(q.res.value)

        return gen_instr(f"MOVE {res_processed},{op_processed} ; ASSIGN op1, res")
    else:
        return ""


def gen_goto(q: Quartet) -> str:
    return ""


def gen_param(q: Quartet) -> str:
    return ""


def gen_return(q: Quartet) -> str:
    return ""


def gen_call(q: Quartet) -> str:
    return ""


code_gen_dict: dict[Operation, Callable[[Quartet], str]] = {
    Operation.ADD: gen_add,
    Operation.OR: gen_or,
    Operation.EQUALS:  gen_equals,
    Operation.ASSIGN:  gen_assign,
    Operation.GOTO:  gen_goto,
    Operation.PARAM:  gen_param,
    Operation.RETURN:  gen_return,
    Operation.CALL:  gen_call,
}


class CodeGenException(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return self.msg
