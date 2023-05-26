import os
import sys
from io import TextIOWrapper


class JspdlProcessor:
    """
    A class that represents a JspdlProcessor.

    Attributes
    ----------
    input : Input
        An instance of the Input class.
    output : Output
        An instance of the Output class.
    lexer : Lexer
        An instance of the Lexer class.
    syntax_semantic_analyzer : SyntaxSemanticAnalyzer or None
        An instance of the SyntaxSemanticAnalyzer class or None.
    compiler : Compiler or None
        An instance of the Compiler class or None.

    Methods
    -------
    __init__()
        Initializes a JspdlProcessor instance and sets up the necessary components.
    """

    class Input:
        """
        A class that represents the input for JspdlProcessor.

        Attributes
        ----------
        input_file : file object
            The input file opened for reading.

        Methods
        -------
        __init__()
            Initializes an Input instance and opens the input file.
        """

        def __init__(self):
            """
            Initializes an Input instance and opens the input file.
            """
            self.input_file = open(sys.argv[1], "r")

    class Output:
        """
        A class that represents the output for JspdlProcessor.

        Attributes
        ----------
        output_dir : str
            The output directory path.
        input_file : file object
            The input file opened for reading.
        files : dict
            A dictionary mapping file names to file objects.

        Methods
        -------
        __init__()
            Initializes an Output instance and creates the necessary output files.
        create_output_files()
            Creates the output files inside the output directory.
        """

        def __init__(self) -> None:
            """
            Initializes an Output instance and creates the necessary output files.
            """
            input_file = open(sys.argv[1], "r")
            input_file_name = os.path.basename(input_file.name)
            self.output_dir = os.path.join(
                os.getcwd(), input_file_name.split(".")[0])
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
            self.input_file = input_file
            self.files = self.Files(self.output_dir)

        class Files:
            def __init__(self, output_dir) -> None:
                self.token_file = open(os.path.join(
                    output_dir, "tokens.txt"), "w")
                self.errors_file = open(os.path.join(
                    output_dir, "errors.txt"), "w")
                self.symbol_table_file = open(os.path.join(
                    output_dir, "symbol_table.txt"), "w")
                self.parse_file = open(os.path.join(
                    output_dir, "parse.txt"), "w")
                self.three_address_code_file = open(os.path.join(
                    output_dir, "three_address_code.txt"), "w")
                self.assembly_code_file = open(os.path.join(
                    output_dir, "assembly_code.txt"), "w")

    @staticmethod
    def parse_arguments():
        """
        Checks if the given arguments are valid.
        """
        argv_length = len(sys.argv)
        if argv_length < 2 or argv_length > 2:
            print(
                "Invalid number of arguments. Use python jspdl.py path/to/program_to_parse[.txt|.jspdly].\nUse python jspdl.py --help for help. ")
            sys.exit()

        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print(
                "Usage: python jspdl.py path/to/program_to_parse[.txt|.jspdly]")
            sys.exit(0)

    def __init__(self) -> None:
        from lexer import Lexer
        """
        Initializes a JspdlProcessor instance and sets up the necessary components.
        """
        JspdlProcessor.parse_arguments()
        self.input = self.Input()
        self.output = self.Output()
        self.lexer = Lexer(self)
        self.syntax_semantic_analyzer = None
        self.compiler = None


if __name__ == "__main__":
    processor = JspdlProcessor()
    processor.lexer.tokenize()
