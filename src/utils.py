import sys
from io import TextIOWrapper


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Error:
    """
    Global error class used for all parts of the procesor. Each part must define its own error() method
    that creates an Error and adds it to the list of errors of said class
    """

    def __init__(self, error_file: TextIOWrapper, msg: str, origin: str, linea: int, writing_line: str, attr=None):
        """

        :param msg[str]: Error message
        :param origin[str]: string that signalizes from which part of the processor the error comes from
        :param linea[int]: line in that error occurs in
        :param attr[]:
        """
        self.msg = msg
        self.line = linea
        self.att = attr
        self.origin = origin
        self.writing_line = writing_line
        self.error_file = error_file
        self.print()

    def print(self):
        """
        Prints the error through stderr
        """
        error_str = "*" * 75 + \
            f"\n{self.origin} at line {self.line}: \n{self.msg}\n" + \
            "*" * 75 + "\n\n"
        writing_line = "*" * 75 + \
            f"\n{self.origin} at line {self.line}: \n{self.writing_line}\n" + \
            "*" * 75 + "\n\n"
        eprint(Colors.FAIL + error_str + Colors.ENDC)
        self.error_file.write(writing_line)


def eprint(*args, **kwargs):
    """
    Prints the given parameters to stderr using special colors
    :param args:
    :param kwargs:
    """
    print(*args, file=sys.stderr, **kwargs)


def gen_error_line(line, start, end):
    """
    :param line: line from the given input file to get
    :param start: starting column to highlight from
    :param end: ending column to highlight to
    """
    writing_line = [line - 1]
    line = Colors.OKBLUE + LINES[line - 1]
    if line[-1:] != "\n":
        line += "\n"
        writing_line += "\n"
    line = Colors.ENDC + line + Colors.FAIL
    line += " " * start
    writing_line += " " * start + "^" + "~" * (end - 1)
    line += Colors.WARNING + "^" + "~" * (end - 1) + Colors.FAIL
    return line, writing_line
