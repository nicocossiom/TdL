from enum import Enum
# Add Type argument to fix the error
from typing import Dict, List, NamedTuple, Type, Union

from tree_sitter import Node

from ast_util import unwrap_text


class Undefined:
    pass


class DefinedFomOperation:
    pass


UndefinedType: Type[Undefined] = Undefined


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
    JSPDLType.INT: 1,
    JSPDLType.BOOLEAN: 1,
    JSPDLType.STRING: 64,
}


class Entry:
    def __init__(self, type: JSPDLType, node: Node) -> None:
        self.type = type
        self.node = node


VarEntryValType = Union[str, bool, int] | Undefined | DefinedFomOperation


class VarEntry(Entry):
    def __init__(self, type: str, value: VarEntryValType, node: Node):
        super().__init__(JSPDLType(type), node)
        self.value = value
        self.size = size_dict[self.type]
        self.offset = -1


class Argument(NamedTuple):
    type: JSPDLType
    id: str


class FnEntry(Entry):
    def __init__(self, return_type: JSPDLType, arguments: List[Argument], node: Node) -> None:
        super().__init__(JSPDLType.FUNCTION, node)
        self.return_type = return_type
        self.arguments: List[Argument] = arguments
        self.function_name = unwrap_text(node.named_children[0].text)


class SymbolTable:
    def __init__(self) -> None:
        self.entries: Dict[str, Entry] = {}
        self.current_offset = 0

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def __getitem__(self, key: str) -> Entry:
        return self.entries[key]

    def __setitem__(self, key: str, value: Entry) -> None:
        if isinstance(value, VarEntry):
            value.offset = self.current_offset
            self.current_offset += value.size
        self.entries[key] = value

    def __repr__(self) -> str:
        return str(self.entries)

    def __str__(self) -> str:
        return str(self.entries)


# Define the type for the symbol table

# Initialize an empty symbol table
global_symbol_table: SymbolTable = SymbolTable()
current_symbol_table: SymbolTable = global_symbol_table
