from lexer import Error, Lexer, Token, gen_error_line
from symbol_table import SymbolTable

First = {
    'P': ["function", "eof", "let", "if", "do", "id", "return", "print", "input"],
    'B': ["let", "if", "do", "id", "return", "print", "input"],
    "T": ["int", "boolean", "string"],
    "S": ["id", "return", "print", "input"],
    "Sp": ["asig", "parAbierto", "postIncrem"],
    "X": ["id", "parAbierto", "int", "cadena", "true", "false", "lambda"],
    "C": ["let", "if", "id", "return", "print", "input", "do"],
    "L": ["id", "parAbierto", "int", "cadena", "true", "false"],
    "Q": "coma",
    "F": "function",
    "H": ["int", "boolean", "string"],
    "A": ["int", "boolean", "string"],
    "K": "coma",
    "E": ["id", "parAbierto", "int", "cteEnt", "cadena", "true", "false", "and", "or", "lambda", "mayor", "equals"],
    "N": ["id", "parAbierto", "int", "cteEnt", "cadena", "true", "false", "and", "or", "lambda", "mayor", "equals"],
    "Z": ["id", "parAbierto", "int", "cteEnt", "cadena", "true", "false", "mas", "por", "lambda"],
    "O1": ["and", "or", "lambda"],
    "O2": ["equals", "mayor", "lambda"],
    "O3": ["por", "mas", "lambda"],
    "R": ["id", "parAbierto", "cteEnt", "cadena", "true", "false"],
    "Rp": ["parAbierto", "postIncrem", "lambda"],
}

# usamos eof como $ para marcar fin de sentencia admisible
Follow = {
    "O1": ["puntoComa", "parCerrado", "coma"],
    "O2": ["mayor", "equals", "parCerrado", "puntoComa", "coma", "and", "or"],
    "O3": ["mayor", "equals", "parCerrado", "coma", "and", "or", "puntoComa"],
    "X": "puntoComa",
    "C": "llaveCerrado",
    "L": "parCerrado",
    "Q": "parCerrado",
    "H": "parAbierto",
    "A": "parCerrado",
    "K": "parCerrado",
    "Rp": ["or", "and", "mas", "por", "coma", "puntoComa", "parCerrado", "equals", "mayor"],
}
SYMB_OPS_R = {
    "mas": "+",
    "por": "*",
    "and": "&&",
    "asig": "=",
    "mayor": ">",
    "coma": ",",
    "puntoComa": ";",
    "parAbierto": "(",
    "parCerrado": ")",
    "llaveAbierto": "{",
    "llaveCerrado": "}",
    "or": "||",
    "postIncrem": "++"
}


class ProductionObject:
    def __init__(self, **kwargs: object) -> None:
        r"""
        An object representing a Syntactic production rule which holds values for a Semantic functions
        :param \**kwargs:
                See below
            :Keyword Arguments:
            * *tipo* (``str``) --
              Extra stuff
            * *ancho* (``str``) --
            * *tipoRet* (``str``) --

        """
        try:
            self.type = kwargs["tipo"]
        except KeyError:
            self.type = None
        try:
            self.size = kwargs["ancho"]
        except KeyError:
            self.size = None
        try:
            self.ret_type = kwargs["tipoRet"]
        except KeyError:
            self.ret_type = None


class Syntactic:
    # Tablas de Símbolos class (static vars of shared by all members of the class) variables,
    # which will be referred to as self.{} inside the methods, but they're not instance class variables
    general_symbol_table: SymbolTable = None
    current_symbol_table: SymbolTable = None
    TSLIST = []

    def __init__(self, lexer: Lexer) -> None:
        self.index = 0  # indice que apunta al elemento actual de la lista de tokens
        self.current_token: Token = None
        self.token = None
        self.last_token = None
        self.last_actual_token = None
        self.lexer = lexer

    def next(self) -> Token:
        """
        Returns code (str) of the next token from the Lexer and stores the actual token in self
        :return: next Token
        """
        self.last_actual_token = self.current_token
        self.last_token = self.current_token
        self.current_token: Token = self.lexer.tokenize()
        self.index += 1
        if not self.current_token:
            return self.next()
        else:
            self.token = self.current_token.code
            return self.current_token

    def equipara(self, code: str, rule=None) -> bool:
        """
        Compares the given code to the actual token, return status based on said comparison.
        If regla is given means we're checking against a First of current state, rule should be given and if true
        add to parse list. If not rule then we're inside a production hence we know what tokens to expect and can
        error if comparison is false

        :param code: code of token expected in current syntactic position
        :param rule: rule to be added to the parse list
        :return: True if code == current token, else False
        """
        # print(f"equipara({self.token} , {code} )", end="")

        if self.token == code:
            # print("CORRECTO")
            if rule:
                # only add rule when it's first check in a function (has regla), and we're sure it's the correct token
                Syntactic.addParseElement(rule)
            self.next()
            return True
        if not rule:  # after first check (means we're in the middle of a state
            # we expected a certain token but it was not it, now we can say it's an error
            if self.token == "eof":
                self.last_actual_token = TOKENLIST[-2]
                try:
                    symbol = SYMB_OPS_R[code]
                except KeyError:
                    symbol = code
                self.error("WrongTokenError", f"No ha cerrado con {symbol}")
            try:
                symbol = SYMB_OPS_R[code]
            except KeyError:
                symbol = code
            self.error("WrongTokenError", f"Recibido {symbol} - Esperaba el token {self.token}",
                       True)
        # print("INCORRECTO -> siguiente")
        return False

    def equierror(self, expected):
        expected_with_symbol = []
        try:
            symbol = SYMB_OPS_R[self.token]
            for elem in expected:
                try:
                    expected_with_symbol.append(SYMB_OPS_R[elem])
                except KeyError:
                    expected_with_symbol.append(elem)

        except KeyError:
            symbol = self.token
        self.error("WrongTokenError",
                   f"Recibido \"{symbol}\" - Esperaba uno de los siguientes tokens {expected_with_symbol}", True)

    def error(self, error_type, msg: string, attr=None):
        if self.token == "eof":
            self.last_actual_token = TOKENLIST[-2]
        token = self.current_token if attr else self.last_actual_token
        strings = gen_error_line(token.line, token.startCol, token.endCol)
        Error(strings[0] + "\n" + msg, error_type, token.line,
              strings[1] + "\n" + msg, attr)
        sys.exit("Error fatal, saliendo ...")

    @staticmethod
    def addParseElement(regla: int) -> None:
        """Writes the given parse element int the tokens.txt """
        global PARSESTRING
        if PARSESTRING is None:
            PARSESTRING = f"Descendente {regla} ".replace("None", "")
        else:
            PARSESTRING += f"{regla} "
        # print(PARSESTRING)

    def start(self):
        """Starts the Syntactic process"""
        self.general_symbol_table = SymbolTable()
        self.TSLIST.append(self.general_symbol_table)
        self.current_symbol_table = self.general_symbol_table
        self.next()
        self.P()
        PARSEFILE.write(PARSESTRING)
        print(Colors.OKGREEN + f"Archivo {sys.argv[1]} analizado, es correcto" + Colors.ENDC
              + "\nErrores corregidos durante el análisis:")

    def P(self) -> None:
        if self.token in First["B"]:
            Syntactic.addParseElement(1)
            self.B()
            self.P()
        elif self.token in First['F']:
            Syntactic.addParseElement(2)
            self.F()
            self.P()
        elif self.equipara("eof"):
            Syntactic.addParseElement(3)
            self.writeTS()
            return

    def B(self) -> ProductionObject(tipo=True):
        if self.equipara("let", 4):
            T = self.T()
            id = self.current_token.att
            if self.equipara("id"):
                if self.equipara("puntoComa"):
                    if not self.current_symbol_table.search_id(id):
                        self.current_symbol_table.insert_id(id, T.type)
                    return
        elif self.equipara("if", 5) and self.equipara("parAbierto"):
            E = self.E()
            if not E:
                self.error("EmptyConditionError", "La condición está vacía")
            if self.equipara("parCerrado"):
                if self.token == "llaveAbierto":
                    self.error("IfBlockError",
                               "Solo estan soportados los ifs simples")
                self.S()
                if E.type != "boolean":
                    self.error("WrongDataTypeError",
                               "El tipo de E tiene que ser boolean ya que nos encontramos en la condición de if")
        elif self.token in First["S"]:
            Syntactic.addParseElement(6)
            self.S()
        elif self.equipara("do", 7):
            if self.equipara("llaveAbierto"):
                self.C()
                if self.equipara("llaveCerrado") and self.equipara("while") and self.equipara("parAbierto"):
                    E = self.E()
                    if not E:
                        self.error("EmptyConditionError",
                                   "La condición está vacía")
                    if self.equipara("parCerrado") and self.equipara("puntoComa"):
                        if E.type != "boolean":
                            self.error(
                                "WrongDataTypeError", "La condición del while debe ser de tipo booleano")
        else:
            self.equierror(First["B"])

    def T(self) -> ProductionObject:
        if self.equipara("int", 8):
            return ProductionObject(tipo="int", ancho=1)
        elif self.equipara("boolean", 9):
            return ProductionObject(tipo="boolean", ancho=1)
        elif self.equipara("string", 10):
            return ProductionObject(tipo="string", ancho=64)
        self.error(
            "TypeError", f"Tipo {self.token} no reconocido, tipos disponibles {First['T']}", True)

    def S(self) -> ProductionObject:
        id = self.current_token.att
        if self.equipara("id", 11):
            Sp = self.Sp()
            if not Sp or isinstance(Sp.type, list):  # es una llamada a funcion
                if self.general_symbol_table.search_id(id):
                    id: SymbolTable.FunctionElement = self.general_symbol_table.map[id]
                    params = id.tipo_params
                    if Sp:
                        params_dados = Sp.type
                        if id.tipo == "function" and params != params_dados:  # funcion con parametros incorrectos
                            self.error("ArgumentTypeError", f"La funcion {id} recibe los argumentos de tipo "
                                                            f"{params}, tipos recibidos {params_dados}")
                else:
                    self.error(
                        "NonDeclaredError", f"Error la función {id} no ha sido declarada previamente")
            else:  # es una asignacion
                var_tabla = None
                if self.current_symbol_table.search_id(id):
                    var_tabla = self.current_symbol_table.map[id]
                elif self.general_symbol_table.search_id(id):
                    var_tabla = self.general_symbol_table.map[id]
                # declaracion e inicialización de una variable global i.e (a = 5)
                if not var_tabla:
                    if Sp.type == "int":
                        id: SymbolTable.SymbolTableElement = self.general_symbol_table.insert_id(
                            id, Sp.type)
                        if Sp.type != "postIncrem" and id.tipo != Sp.type:
                            self.error("TypeError",
                                       f"Tipo de la variable: \"{id.lex}\" no coincide con tipo tipo de la asignación: \"{Sp.type}\"")
                        return ProductionObject(tipo=True)
                    else:
                        self.error(
                            "TypeError", f"Solo se pueden hacer asignaciones sin inicializacion cuando la asignacion es de tipo int, en este caso es de tipo {Sp.type}")
                elif Sp.type == "postIncrem" and var_tabla.tipo != "int":
                    self.error("WrongDataTypeError",
                               "El operador post incremento solo es aplicable a variables del tipo entero")
                elif Sp.type != "postIncrem" and var_tabla.tipo != Sp.type:  # es una asignacion normal
                    self.error(
                        "TypeError", f"Tipo de la variable: \"{var_tabla.tipo}\" no coincide con el tipo de la asignación: \"{Sp.type}\"")

        elif self.equipara("return", 12):
            X = self.X()
            if self.equipara("puntoComa"):
                return ProductionObject(tipo=True, tipoRet=X.type)

        elif self.equipara("print", 13):
            if self.equipara("parAbierto"):
                E = self.E()
                if not E:
                    self.error("EmptyConditionError",
                               "La condición está vacía")
                if self.equipara("parCerrado") and self.equipara("puntoComa"):
                    if E.type in {"string", "int"}:
                        return ProductionObject(tipo=True)
                    else:
                        self.error(
                            "WrongDataTypeError", "La función print solo acepta parámetros de tipo string")

        elif self.equipara("input", 14) and self.equipara("parAbierto"):
            id = self.current_token.att
            if self.equipara("id") and self.equipara("parCerrado") and self.equipara("puntoComa"):
                tipo = None
                if self.current_symbol_table.search_id(id):
                    tipo = self.current_symbol_table.map[id].tipo
                elif self.general_symbol_table.search_id(id):
                    tipo = self.general_symbol_table.map[id].tipo
                if tipo not in {"int", "string"}:
                    self.error("TypeError", f"Variable a es de tipo {tipo}, input() debe recibir una variable de "
                                            f"tipo string o entero")
                if not tipo:
                    self.general_symbol_table.insert_id(id, "int")
                else:
                    return ProductionObject(tipo=True)

        else:
            self.equierror(First["S"])

    def Sp(self) -> ProductionObject:
        if self.equipara("asig", 15):
            E = self.E()
            if not E:
                self.error("EmptyConditionError", "La condición está vacía")
            if self.equipara("puntoComa"):
                return E
        elif self.equipara("parAbierto", 16):
            L = self.L()
            if self.equipara("parCerrado") and self.equipara("puntoComa"):
                if L:
                    return ProductionObject(tipo=L.type)
        elif self.equipara("postIncrem", 17) and self.equipara("puntoComa"):
            return ProductionObject(tipo="postIncrem")
        else:
            self.equierror(First["Sp"])

    def X(self) -> ProductionObject:
        if self.token in First['E']:
            Syntactic.addParseElement(18)
            return self.E()
        elif self.token in Follow['X']:
            Syntactic.addParseElement(19)
            return ProductionObject(tipo=True)
        else:
            self.error("SentenceNotTerminatedError",
                       f"Esperaba ';' al terminar la sentencia, después de un return vacío")

    def C(self) -> None:
        if self.token in First["B"]:
            Syntactic.addParseElement(20)
            if self.token != "eof":
                self.B()
                self.C()
        elif self.token in Follow['C']:
            Syntactic.addParseElement(21)

    def L(self) -> ProductionObject:
        if self.token in First["E"]:
            Syntactic.addParseElement(22)
            E = self.E()
            if E:
                return ProductionObject(tipo=self.Q([E.type]))
            return ProductionObject(tipo=[])
        elif self.token in Follow['L']:
            Syntactic.addParseElement(23)
        else:
            self.error("FunctionCallError",
                       "No se ha cerrado paréntesis en la llamada a la función")

    def Q(self, lista=None) -> List[str]:
        if self.equipara("coma", 24):
            Q = lista if lista else []
            E = self.E()
            if E:
                Q.append(E.type)
                return self.Q(lista)
        elif self.token in Follow['Q']:
            Syntactic.addParseElement(25)
            return lista if lista else None
        self.equierror(First["Q"])

    def F(self) -> ProductionObject:
        if self.equipara("function", 26):
            id = self.current_token.att
            if self.equipara("id"):
                tipo_ret = self.H().type
                self.current_symbol_table = SymbolTable(id)  # tabla de funcion
                self.TSLIST.append(self.current_symbol_table)
                if self.equipara("parAbierto"):
                    A = self.A()
                    if A:
                        tipo_params = A.type
                    else:
                        tipo_params = ""
                    if self.equipara("parCerrado") and self.equipara("llaveAbierto"):
                        self.general_symbol_table.insert_id(
                            id, "funcion", tipo_params, len(tipo_params), tipo_ret)
                        self.C()
                        if self.equipara("llaveCerrado"):
                            # insertar funcion en TSG de una
                            # ~= destruir tabla de la funcion
                            self.current_symbol_table = self.general_symbol_table
                            return ProductionObject(tipo=True)
        else:
            self.equierror(First["F"])

    def H(self) -> ProductionObject:
        if self.token in First['T']:
            Syntactic.addParseElement(27)
            T = self.T()
            return ProductionObject(tipo=T.type)
        elif self.token in Follow['H']:
            Syntactic.addParseElement(28)
            return ProductionObject(tipo="")
        else:
            self.error("TypeError",
                       f"Tipo de función no aceptado. Debe usar {First['T']} o \"\" (no poner nada para void)")

    def A(self) -> ProductionObject:
        if self.token in First['T']:
            Syntactic.addParseElement(29)
            T = self.T()
            id = self.current_token.att
            if self.equipara("id"):
                K = self.K([T.type])
                self.current_symbol_table.insert_id(id, T.type)
                if K:
                    return ProductionObject(tipo=K)
        elif self.token in Follow['A']:
            Syntactic.addParseElement(30)
        else:
            self.error("FunctionCallError",
                       f"No ha cerrado paréntesis en la llamada a la función")

    def K(self, lista=None) -> List[str]:
        if self.equipara("coma", 31):
            K = lista if lista else []
            T = self.T()
            K.append(T.type)
            id = self.current_token.att
            if self.equipara("id"):
                self.current_symbol_table.insert_id(id, T.type)
                return self.K(lista)
        elif self.token in Follow['K']:
            Syntactic.addParseElement(32)
            return lista if lista else None
        else:
            self.error("ArgumentDeclarationError",
                       "Los argumentos de las funciones deben estar separados por \',\'")

    def E(self) -> ProductionObject:
        if self.token in First["N"]:
            Syntactic.addParseElement(33)
            N = self.N()  # primer argumento
            return self.O1(N)

    def N(self) -> ProductionObject:
        if self.token in First["Z"]:
            Syntactic.addParseElement(34)
            Z = self.Z()
            # si llega aqui no ha habido errores entonces devolvemos el tipo que espera O2 por si ha llamado a Z
            return self.O2(Z)

    def Z(self) -> ProductionObject:
        if self.token in First["R"]:
            Syntactic.addParseElement(35)
            R = self.R()
            # si llega aqui no ha habido errores entonces devolvemos el tipo que espera O3 por si ha llamado a R
            return self.O3(R)

    def O1(self, prev=None) -> ProductionObject:
        if self.equipara("or", 36):
            N = self.N()
            if N.type != "boolean":
                self.error(
                    "WrongDataTypeError", f"Operador || solo acepta datos lógicos, tipo dado {N.type}")
            return self.O1()
        elif self.equipara("and", 37):
            N = self.N()
            if N.type != "boolean":
                self.error(
                    "OperandTypeError", f"Operador && solo acepta datos lógicos, tipo dado {N.type}")
            return self.O1()
        elif self.token in Follow['O1']:
            Syntactic.addParseElement(38)
            if prev:
                return prev
            return ProductionObject(tipo="boolean")
        else:
            self.error("NonSupportedOperationError",
                       f"Esperaba uno de los siguientes símbolos{Follow['O1']}")

    def O2(self, prev=None) -> ProductionObject:
        if self.equipara("equals", 39):
            Z = self.Z()
            if Z.type != "int":
                self.error(
                    "OperandTypeError", f"Operador == solo acepta datos de tipo entero, tipo dado {Z.type}")
            return self.O2()
        elif self.equipara("mayor", 40):
            Z = self.Z()
            if Z.type != "int":
                self.error(
                    "OperandTypeError", f"Operador > solo acepta datos de tipo entero, tipo dado {Z.type}")
            return self.O2()
        elif self.token in Follow['O2']:
            Syntactic.addParseElement(41)
            if prev:
                return prev
            return ProductionObject(tipo="boolean")
        else:
            self.error("NonSupportedOperationError",
                       f"Esperaba uno de los siguientes símbolos{Follow['O2']}")

    def O3(self, prev=None) -> ProductionObject:
        if self.equipara("mas", 42):
            R = self.R()
            if R.type != "int":
                self.error(
                    "OperandTypeError", f"Operador + solo acepta datos enteros, tipo dado {R.type}")
            else:
                return self.O3()
        elif self.equipara("por", 43):
            R = self.R()
            if R.type != "int":
                self.error(
                    "OperandTypeError", f"Operador * solo acepta datos enteros, tipo dado {R.type}")
            return self.O3()
        elif self.token in Follow['O3']:
            Syntactic.addParseElement(44)
            if prev:
                return prev
            return ProductionObject(tipo="int")
        else:
            self.error("NonSupportedOperationError",
                       f"Esperaba uno de los siguientes símbolos{Follow['O3']}")

    def R(self) -> ProductionObject:
        id = self.current_token.att
        if self.equipara("id", 45):
            Rp = self.Rp()
            if Rp:  # es una llamada a una funcion o post incremento
                if Rp.type == "postIncrem":
                    if not self.general_symbol_table.search_id(id) and not self.current_symbol_table.search_id(id):
                        self.error(
                            "NonDeclaredError", f"Error la variable {id} no ha sido declarada previamente")
                    try:
                        ident = self.current_symbol_table.map[id]
                    except KeyError:
                        ident = self.general_symbol_table.map[id]
                    if ident.tipo != "int":
                        self.error("OperandTypeError",
                                   "El operador post incremento solo es aplicable a variables del tipo entero")
                elif not self.general_symbol_table.search_id(id):
                    self.error(
                        "NonDeclaredError", f"Error la función {id} no ha sido declarada previamente")
                elif Rp.type != "true" and Rp.type != self.general_symbol_table.map[id].tipo_params:
                    self.error(
                        "WrongArguemensError", f"Tipos de los atributos incorrectos en llamada a función \"{id}\" ")
                else:
                    return ProductionObject(tipo=self.general_symbol_table.map[id].tipo_dev)
            else:
                if self.current_symbol_table.search_id(id):
                    return ProductionObject(tipo=self.current_symbol_table.map[id].tipo)
                if self.general_symbol_table.search_id(id):
                    return ProductionObject(tipo=self.general_symbol_table.map[id].tipo)
                else:
                    self.error(
                        "NonDeclaredError", f"Error la variable {id} no ha sido declarada previamente")
        if self.equipara("parAbierto", 46):
            E = self.E()
            if self.equipara("parCerrado"):
                return ProductionObject(tipo=E.type)
        elif self.equipara("cteEnt", 47):
            return ProductionObject(tipo="int", ancho=1)
        elif self.equipara("cadena", 48):
            return ProductionObject(tipo="string", ancho=1)
        elif self.equipara("true", 49):
            return ProductionObject(tipo="boolean", ancho=1)
        elif self.equipara("false", 50):
            return ProductionObject(tipo="boolean", ancho=1)

    def Rp(self) -> ProductionObject:
        if self.equipara("parAbierto", 51):
            L = self.L()
            if self.equipara("parCerrado"):
                if L:
                    return ProductionObject(tipo=L.type)
                else:
                    return ProductionObject(tipo="true")
        elif self.equipara("postIncrem", 52):
            return ProductionObject(tipo="postIncrem")
        elif self.token in Follow["Rp"]:
            Syntactic.addParseElement(53)

    def writeTS(self):
        for ts in self.TSLIST:
            TSFILE.write(str(ts))
