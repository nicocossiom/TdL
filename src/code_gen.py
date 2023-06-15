from enum import Enum
from typing import Callable, Optional

static_memory_start = 0

static_memory_end = 0

code_start = """
ORG 
"""


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


Operand = tuple[int | str | bool, OperandScope]
OpResult = int | str | bool | None
Quartet = tuple[Operation, Optional[Operand],
                Optional[Operand], Optional[OpResult], ]

# queue of quartets to be processed after code is type-checked -> all quartets are generated
quartet_queue: list[Quartet] = []


def count_global_memory():
    for q in quartet_queue:
        if q[1] and q[1][1] == OperandScope.GLOBAL:
            global static_memory_end
            static_memory_end += 1


def gen_code():
    count_global_memory()
    print(static_memory_end)
    for q in quartet_queue:
        # call the function that generates the code for the operation
        print(q)
        code_gen_dict[q[0]](q)


def gen_add(q: Quartet):
    pass


def gen_or(q: Quartet):
    pass


def gen_equals(q: Quartet):
    pass


def gen_assign(q: Quartet):
    pass


def gen_goto(q: Quartet):
    pass


def gen_param(q: Quartet):
    pass


def gen_return(q: Quartet):
    pass


def gen_call(q: Quartet):
    pass


code_gen_dict: dict[Operation, Callable[[Quartet], None]] = {
    Operation.ADD: gen_add,
    Operation.OR: gen_or,
    Operation.EQUALS:  gen_equals,
    Operation.ASSIGN:  gen_assign,
    Operation.GOTO:  gen_goto,
    Operation.PARAM:  gen_param,
    Operation.RETURN:  gen_return,
    Operation.CALL:  gen_call,
}
