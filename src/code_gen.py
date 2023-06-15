from enum import Enum
from typing import Callable, Optional

static_memory_start = 0

static_memory_current_offset = static_memory_start

static_memory_end = static_memory_start


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

    def __repr__(self) -> str:
        return self.name


OpResult = int | str | bool | None


class Operand:
    def __init__(self, offset: int, scope: OperandScope):
        self.offset = offset
        self.scope = scope

    def __repr__(self) -> str:
        return f"[offset={self.offset},{self.scope}]"


class Quartet:
    def __init__(self, op: Operation, op1: Operand | None = None, op2: Operand | None = None, res: OpResult = None) -> None:
        self.op: Operation = op
        self.op1: Optional[Operand] = op1
        self.op2: Optional[Operand] = op2
        self.res: Optional[OpResult] = res

    def __repr__(self) -> str:
        return f"({self.op}, op1={self.op1}, op2={self.op2}, res={self.res})"


# queue of quartets to be processed after code is type-checked -> all quartets are generated
quartet_queue: list[Quartet] = []


def count_global_memory():
    for q in quartet_queue:
        if q.op1 and q.op1.scope == OperandScope.GLOBAL:
            global static_memory_end
            static_memory_end += 1


def gen_code():
    count_global_memory()
    print(f"Static memory needed to be allocated = {static_memory_end}")
    print("Assembly code:")
    assembly = """
;-----------------------------------------------------------------------------------------
                        ORG 0
                        MOVE #static_memory_start, .IY
"""
    print(f"initial assembly: \n{assembly}")
    print("Quartets generated:")
    for q in quartet_queue:
        # call the function that generates the code for the operation
        print(f"\t{q}")
        assembly += code_gen_dict[q.op](q)
    assembly += f"""
                        HALT
static_memory_start:    RES {static_memory_end-static_memory_start} ; reserve {static_memory_current_offset} memory addresses for global variables
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
    return ""


def gen_or(q: Quartet) -> str:
    return ""


def gen_equals(q: Quartet) -> str:
    return ""


def gen_assign(q: Quartet) -> str:
    global static_memory_current_offset
    if not q.op1 or not q.res:
        raise Exception(
            "Assign operation must have at least one operand and a result")

    if q.op1.scope == OperandScope.GLOBAL:
        static_memory_current_offset += q.op1.offset
        print(f"\tassign: dir 0x{static_memory_current_offset} = {q.res}")
        return gen_instr(f"MOVE #{q.res},#{static_memory_current_offset-1}[.IY] ; ASSIGN op1, res")
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
