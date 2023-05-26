class SymbolTable:
    CREATION_COUNTER = 0

    def __init__(self, name=None):
        self.map = {}
        self.name = "TSG" if name is None else name
        self.pos = 0
        self.creation_number = SymbolTable.CREATION_COUNTER
        SymbolTable.CREATION_COUNTER += 1

    def write_to_st(self):
        """
        Writes the TS (self) in th the TS.txt file in the output directory
        :return:
        """
        TSFILE.write(str(self))

    def __str__(self):
        size_sep = 75
        name = "TABLA PRINCIPAL" if self.name == "TSG" else f"TABLA de funcion \"{self.name}\""
        final_string = "\n" + "-" * size_sep + \
            f"\n\t\t\t{name} #{self.creation_number}\n"
        for lex, entrada in self.map.items():
            final_string += f"\n*  LEXEMA : \"{lex}\"" \
                            f"\n   ATRIBUTOS : \n\t\t" \
                            f"+ Tipo: {entrada.tipo}\n"
            if isinstance(entrada, SymbolTable.FunctionElement):
                final_string += f"\t\t+numParam: {entrada.num_param}\n\t\t\t"

                for i in range(len(entrada.tipo_params)):
                    final_string += f"+ TipoParam{i}: {entrada.tipo_params[i]}\n\t\t\t"

                final_string += f"+TipoRetorno: {entrada.tipo_dev}\n"
            else:
                final_string += f"\t\t+ Despl: {entrada.desp}\n"
        return final_string + "-" * size_sep

    @staticmethod
    def get_desp(tipo):
        """
        Given a type returns its value for size
        :param tipo: type whose size we want to know
        :return: size of tipo in Bytes
        """
        res = 0  # function
        if tipo == "boolean":
            res = 1
        elif tipo == "int":
            res = 1
        elif tipo == "string":
            res = 8
        return res

    def search_id(self, given_id: str):
        """
        Searches for an id in the table
        :param given_id: id we want to check
        :return: True if found, False if not
        """
        try:
            self.map[given_id]
        except KeyError:
            return False
        return True

    def insert_id(self, given_id: str, tipo: str, *args):
        """

        :param given_id:
        :param tipo:
        :param args:
        :return:
        """
        if not self.search_id(given_id):
            if len(args) == 0:
                elem = SymbolTable.SymbolTableElement(self, given_id, tipo)
            else:
                elem = SymbolTable.FunctionElement(self, given_id, tipo, args)
            self.map[given_id] = elem
            return elem
        else:
            raise Exception("Identificador ya existe en la TS actual")

    class SymbolTableElement:
        def __init__(self, ts, identifier: str, tipo: str):
            """
            :param identifier:
            :param tipo:
            """
            self.ts = ts
            self.lex = identifier
            self.tipo = tipo
            self.desp = self.ts.pos
            self.ts.pos += SymbolTable.get_desp(tipo)

    class FunctionElement(SymbolTableElement):
        def __init__(self, *args):
            """
            :param identifier:
            :param tipo
            :param *args:
                See below
            *param 1(List[str]): tipo_params
            """
            super().__init__(args[0], args[1], args[2])
            self.tipo_params = [elem for elem in args[3][0]]
            self.num_param = args[3][1]
            self.tipo_dev = args[3][2]
