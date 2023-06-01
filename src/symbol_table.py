from enum import Enum
from typing import Dict, List, NamedTuple, Optional, TypedDict, Union

from tree_sitter import Node


class JSPDLType(Enum):
    UNBOUND = "unbound"
    INT = "int"
    BOOLEAN = "boolean"
    STRING = "string"


size_dict: Dict[JSPDLType, int] = {
    JSPDLType.INT: 4,
    JSPDLType.BOOLEAN: 1,
    JSPDLType.STRING: 1,
}

size_dict = {
    JSPDLType.INT: 4,
    JSPDLType.BOOLEAN: 1,
    JSPDLType.STRING: 1,
}


class VarEntry:
    def __init__(self, type: str, value: Optional[Union[str, bool, int]], offset: Optional[int], node: Node):
        self.type: JSPDLType = JSPDLType(type)
        self.value = value
        self.offset = offset
        self.size = size_dict[self.type]
        self.node = None


class Argument(NamedTuple):
    type: str
    id: str


class FnEntry(TypedDict):
    return_type: str
    arguments: Optional[List[Argument]]
    node: Node


# Define the type for the symbol table
SymbolTable = Dict[str, Union[VarEntry, FnEntry]]

# Initialize an empty symbol table
symbol_table: SymbolTable = {}
current_symbol_table: SymbolTable = symbol_table
