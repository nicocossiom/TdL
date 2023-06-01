"""
    Build the JSPDL language library using Tree-sitter and create the parser with 
    the language set.
"""
from tree_sitter import Language, Parser

Language.build_library(
    # Store the library in the `build` directory
    'jspdl_build/jspdl.so',

    # Include one or more languages
    [
        'jspdl_language',
    ]
)

language = Language('jspdl_build/jspdl.so', 'jspdl')
parser = Parser()
parser.set_language(language)  # parser used for parsing
