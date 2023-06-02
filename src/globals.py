from io import TextIOWrapper
from typing import List, LiteralString, Optional

# file being analyzed (relative path)
file: Optional[TextIOWrapper] = None
# file being analyzed (full path)
file_path = None
# file lines (list of strings)
file_lines: Optional[List[str]] = None
# base name of file
file_name: Optional[str] = None
# code being analyzed (string)
code = None
