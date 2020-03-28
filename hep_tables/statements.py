# Code to implement statments that will build up the LINQ query, a bit at a time.
import ast
from hep_tables.utils import new_var_name
from typing import Type, List

from func_adl import ObjectStream


class statement_base:
    '''
    Base statement. Should never be created directly.
    '''
    def __init__(self, ast_rep: ast.AST, rep_type: Type):
        self._ast = ast_rep
        self.rep_type = rep_type

    def apply(self, stream: object) -> object:
        assert False, 'This should be overridden'

    def apply_as_text(self, var_name: str) -> str:
        assert False, 'This should be overriden'

    def apply_as_function(self, var_name: str) -> str:
        assert False, 'This should be overriden'


class statement_unwrap_list(statement_base):
    '''
    A placeholder statement. Used to unwrap a type
    '''
    def __init__(self, ast_rep: ast.AST, rep_type: Type):
        assert rep_type is List[object]
        statement_base.__init__(self, ast_rep, object)


class statement_df(statement_base):
    '''
    Represents the dataframe/EventDataSet that is the source of all data.
    '''
    def __init__(self, ast_rep: ast.AST):
        statement_base.__init__(self, ast_rep, object)


class statement_select(statement_base):
    '''
    Represents a transformation or mapping. Two types are handled:

        - Object transformation: df -> df.Select(lambda e: e.jets())
        - Sequence of objects tranformation:
            df -> df.Select(lambda e1: e1.Select(lambda e2: e2.jets()))
    '''
    def __init__(self, ast_rep: ast.AST, rep_type: Type, var_name: str,
                 function_text: str, is_sequence_of_objects: bool):
        statement_base.__init__(self, ast_rep, rep_type)
        self._iterator = var_name
        self._func = function_text
        self._act_on_sequence = is_sequence_of_objects

    def apply(self, seq: object) -> ObjectStream:
        # Build the lambda
        assert isinstance(seq, ObjectStream), 'Internal error'
        if self._act_on_sequence:
            outter_var_name = new_var_name()
            lambda_text = f'lambda {outter_var_name}: {outter_var_name}' \
                f'.Select(lambda {self._iterator}: {self._func})'
        else:
            lambda_text = f'lambda {self._iterator}: {self._func}'

        return seq.Select(lambda_text)

    def apply_as_text(self, var_name: str) -> str:
        # BUild the lambda, but all in text
        if self._act_on_sequence:
            outter_var_name = new_var_name()
            lambda_text = f'lambda {outter_var_name}: {outter_var_name}' \
                f'.Select(lambda {self._iterator}: {self._func})'
        else:
            lambda_text = f'lambda {self._iterator}: {self._func}'

        return f'{var_name}.Select({lambda_text})'

    def apply_as_function(self, var_name: str) -> str:
        if self._act_on_sequence:
            inner_var = new_var_name()
            inner_expr = self._func.replace(self._iterator, inner_var)
            expr = f'{var_name}.Select(lambda {inner_var}: {inner_expr})'
            return expr
        else:
            return self._func.replace(self._iterator, var_name)


class statement_where(statement_base):
    '''
    Represents a filtering. Two forms are handled.
        - Object filter: df -> df.Where(lambda e: e.jets())
        - Sequence of objects filter:
            df -> df.Select(lambda e1: e1.Where(lambda e2: e2.jets()))
    '''
    def __init__(self, ast_rep: ast.AST, rep_type: Type, var_name: str,
                 function_text: str, is_sequence_of_objects: bool):
        statement_base.__init__(self, ast_rep, rep_type)
        self._iterator = var_name
        self._func = function_text
        self._act_on_sequence = is_sequence_of_objects

    def apply(self, seq: object) -> ObjectStream:
        # Build the lambda
        assert isinstance(seq, ObjectStream), 'Internal error'
        if self._act_on_sequence:
            outter_var_name = new_var_name()
            lambda_text = f'lambda {outter_var_name}: {outter_var_name}' \
                f'.Where(lambda {self._iterator}: {self._func})'
            return seq.Select(lambda_text)
        else:
            lambda_text = f'lambda {self._iterator}: {self._func}'
            return seq.Where(lambda_text)


class statement_constant(statement_base):
    '''
    A bit of a weird one - returns  a constant that should be used
    directly as input for the next thing.
    '''
    def __init__(self, ast_rep: ast.AST, value: object, rep_type: Type):
        statement_base.__init__(self, ast_rep, rep_type)
        self._value = value

    def apply(self, stream: object) -> object:
        return self._value
