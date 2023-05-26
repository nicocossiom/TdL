import string
from io import TextIOWrapper

from utils import Colors, Error, eprint, gen_error_line


class Lexicon:
    RESERVED_WORDS = ["let", "function", "rn", "else", "input", "print", "while", "do", "true", "false", "int", "boolean",
                      "string", "return", "if"],
    LETTERS = "abcdefghijlmnñopqrstuvwxyzABCDEFGHIJKLMNÑOPQRSTUVWXYZ",
    SYMB_OPS = {
        "+": "mas",
        "*": "por",
        "&": "and",
        "=": "asig",
        ">": "mayor",
        ",": "coma",
        ";": "puntoComa",
        "(": "parAbierto",
        ")": "parCerrado",
        "{": "llaveAbierto",
        "}": "llaveCerrado",
        "|": "or"
    }


class Token:
    """A token representation for the processor"""

    def __init__(self, code: str, line, start_col, end_col, attribute=None):
        """
        :param line:
        :param start_col:
        :param end_col:
        :param attribute:
        """
        self.code = code
        self.att = attribute
        self.line = line
        self.start_col = start_col
        self.end_col = end_col

    def print(self):
        print(
            f"{self.code},{self.att} line: {self.line}, cols: {self.start_col} -> {self.end_col}")


class Lexer:
    from jspdl import JspdlProcessor

    def __init__(self, jspdl_processor: JspdlProcessor):
        self.num = 0  # current integer number being constructed
        self.lex = ""  # current string being constructed
        self.char = ""  # current character being read
        self.line = 1
        self.tokenizing = True  # keeps track of if a token is being built -> True:yes, False:No
        self.col = 1
        self.start_col = 0
        self.jspdl_processor = jspdl_processor
        self.input_file = self.jspdl_processor.input.input_file
        self.token_list = []

    def skip_block_comments(self):
        """Skips block comments and detects error in its specification"""
        self.next_char()
        if self.char == "*":
            self.next_char()
            while self.peek_char() != "/" and self.char != "*" and self.char != "":
                self.next_char()
                if self.char == "":
                    self.error("Comentario de bloque no cerrado")
                if self.char == "\n":
                    self.line += 1
                    self.start_col = 0
                    self.col = 1
            self.next_char()
            self.next_char()
        elif self.char == "/":
            self.input_file.readline()
            self.col = 1
            self.next_char()
            self.error("Comentarios de tipo '//comentario' no estan permitidos")
            self.line += 1

        else:
            self.error(f"Simbolo {self.char} no pertenece al lenguaje")

    def next_char(self):
        """
        Reads the next character from the file while augmenting the column counter at the same time
        :return:
        """
        self.char = self.input_file.read(1)
        self.col += 1

    def skip_delimeters(self):
        """Skips delimiters such as \\t and \\n """
        if self.char != "":
            while self.char != "" and ord(self.char) < 33:
                if self.char == "\n":
                    self.line += 1
                    self.col = 1
                    self.start_col = 0
                if self.char == "":
                    break  # Block comment processing
                self.next_char()
            if self.char == "/":
                self.skip_block_comments()
                self.skip_delimeters()

    def next(self):
        """Retrieves next character recognized in the language for processing"""
        self.next_char()
        if self.char != "":
            if self.char != "/":
                self.skip_delimeters()
            else:
                self.skip_block_comments()
                self.skip_delimeters()

    def generate_number(self):
        """Adds current char to number in construction after multiplying the number by 10 """
        self.num = self.num * 10 + int(self.char)

    def concatenate(self):
        """Concatenates current char to lexeme in contruction"""
        self.lex += self.char

    def write_token(self, given_token: Token):
        """Writes the given token in the token.txt file \n
        Format:  < code , [attribute] >
        """
        #
        self.jspdl_processor.output.files.token_file.write(
            f"< {given_token.code} , {given_token.att} >\n")

    def peek_char(self) -> str:
        """Returns the character next to that which the file pointer is at, without advancing said file pointer"""
        pos = self.input_file.tell()
        car = self.input_file.read(1)
        self.input_file.seek(pos)
        return car

    def error(self, msg: str, attr=None):
        self.tokenizing = False
        strings = gen_error_line(self.line, self.start_col, self.col)
        Error(self.jspdl_processor.output.files.errors_file,
              strings[0] + "\n" + msg, "Lexical error",
              self.line, strings[1] + "\n" + msg, attr)

    # < codigo , atributo >
    def gen_token(self, code: str, attribute=None) -> Token:
        """Generates a token and appends it to the list of tokens:\n
        -code: specifies token code (id,string,cteEnt,etc)
        -attribute: (OPTIONAL) specifies an attribute if the token needs it i.e < cteEnt , valor >
        """
        self.tokenizing = False
        generated_token = Token(
            code, self.line, self.start_col, self.col, attribute)

        self.token_list.append(generated_token)
        self.write_token(generated_token)
        self.lex = ""
        return generated_token

    def get_quotation(self):
        """
        Looks for \" inside a string, if found it concatenates " to the string and advances to
        the next (not looked at) character and tries to find another following quotation
        :return: False if there is no next quotation
        """
        if self.char != "" and self.char == "\\" and (self.peek_char() == '\"' or self.peek_char() == "\'"):
            self.next_char()
            self.concatenate()
            self.next_char()
            return True if self.get_quotation() else False
        else:
            return False

    def tokenize(self):
        """
        Generates and returns a token if found, and lexical errors if any occur.
        """
        self.tokenizing = True
        if self.peek_char() == "":
            return self.gen_token("eof")
        self.start_col = self.col

        while self.tokenizing:
            self.next()
            self.process_character()
        # remove the last token created after stopping tokenizing
        return self.token_list[-1]

    def process_character(self):
        if self.char.isdigit() and self.lex == "":
            self.generate_number()
            if not self.peek_char().isdigit() and self.num < 32768:
                self.num = 0
                return self.gen_token("cteEnt", self.num)
            else:
                self.error(
                    "Digito con valor mayor al permitido (32768) en el sistema")

        if self.char in Lexicon.LETTERS or self.lex != "":
            return self.process_identifier()

        if self.char == "\"" or self.char == "\'":
            return self.process_string()

        if self.char in Lexicon.SYMB_OPS:
            return self.process_operator()

        self.error(f"Simbolo: \"{self.char}\" no permitido. \nNo pertence al lenguaje, consulte la documentacion para "
                   f"ver carácteres aceptados")

        while self.char != "":
            self.tokenize()

        return None

    def process_identifier(self):
        result = None
        self.concatenate()
        next_car = self.peek_char()

        if not next_car.isdigit() and next_car not in Lexicon.LETTERS and next_car != "_" or next_car == "":
            if self.lex in Lexicon.LETTERS:
                result = self.gen_token(self.lex)
            else:
                if len(self.lex) < 65:
                    result = self.gen_token("id", self.lex)
                else:
                    self.error(
                        f"Identificador {self.lex} excede el tamaño máximo de caracteres permitido (64)")

        return result

    def process_string(self):
        result = None
        if self.char == "\'":
            self.error(
                "Cadena se debe especificar entre \" \", no con \' \'. Corregido")
        self.next()
        while self.char != "" and not self.get_quotation() and (self.char != "\"" and self.char != '\''):
            self.concatenate()
            self.next_char()
        if self.char == "":
            self.error("Cadena debe ir entre \", falta el segundo")
        elif len(self.lex) < 65:
            result = self.gen_token("cadena", self.lex)
        else:
            self.error(
                "Cadena excede el tamaño máximo de caracteres permitido (64)")

        return result

    def process_operator(self):
        result = None
        if self.char == "+":
            if self.peek_char() == "+":
                result = self.gen_token("postIncrem")
                self.next()
            else:
                result = self.gen_token("mas")
        elif self.char == "&" and self.peek_char() == "&":
            result = self.gen_token("and")
            self.next()
        elif self.char == "|" and self.peek_char() == '|':
            result = self.gen_token("or")
            self.next()
        elif self.char == "=":
            if self.peek_char() == "=":
                result = self.gen_token("equals")
                self.next()
            else:
                result = self
        return result

    def tokenize_all(self):
        """
        Goes through the whole file, tokenizing each character
        """
        self.char = self.input_file.read(1)
        while self.char != "":
            self.tokenize()
