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

    def apply(self, stream: ObjectStream) -> ObjectStream:
        assert False, 'This should be overridden'

    def apply_as_text(self, var_name: str) -> str:
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

    def apply(self, seq: ObjectStream) -> ObjectStream:
        # Build the lambda
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


class seq_info:
    '''
    Contains the info for the sequence at its current state:

        - The element that will move things forward
        - The type. The type is the element of the sequence. So the top level event is an
          Event object, the jets in the event are a list of jets... so think of the type as
          one level down, like the template type of a monad.
    '''
    def __init__(self, functor_linq_phrase, t: Type):
        '''
        Arguments:
            functor_linq_phrase         The functor you apply to a LINQ expression to drive this
                                        bit forward
            t                           The type
        '''
        self.functor = functor_linq_phrase
        self.type = t
