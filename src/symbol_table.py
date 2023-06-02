from enum import Enum
from typing import Dict, List, NamedTuple, Optional, TypedDict, Union

from tree_sitter import Node

from ast_util import unwrap_text


class JSPDLType(Enum):
    FUNCTION = "function"
    UNBOUND = "unbound"
    INVALID = "invalid"
    INT = "int"
    BOOLEAN = "boolean"
    STRING = "string"
    VOID = "void"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


size_dict: Dict[JSPDLType, int] = {
    JSPDLType.INT: 4,
    JSPDLType.BOOLEAN: 1,
    JSPDLType.STRING: 1,
}


class Entry:
    def __init__(self, type: JSPDLType, node: Node) -> None:
        self.type = type
        self.node = node


class VarEntry(Entry):
    def __init__(self, type: str, value: Optional[Union[str, bool, int]], offset: Optional[int], node: Node):
        super().__init__(JSPDLType(type), node)
        self.value = value
        self.offset = offset


class Argument(NamedTuple):
    type: JSPDLType
    id: str


class FnEntry(Entry):
    def __init__(self, return_type: JSPDLType, arguments: List[Argument], node: Node) -> None:
        super().__init__(JSPDLType.FUNCTION, node)
        self.return_type = return_type
        self.arguments: List[Argument] = arguments
        self.function_name = unwrap_text(node.named_children[0].text)


# Define the type for the symbol table
SymbolTable = Dict[str, FnEntry | VarEntry]

# Initialize an empty symbol table
symbol_table: SymbolTable = {}
current_symbol_table: SymbolTable = symbol_table
