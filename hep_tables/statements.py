from __future__ import annotations

import ast
import re
from typing import List, Tuple, Type, Union

from func_adl import ObjectStream

from hep_tables.utils import _index_text_tuple, new_var_name, _is_list, _unwrap_list


def _unwrap_list_df(s: statement_base) -> Type:
    if isinstance(s, statement_df):
        return object
    return _unwrap_list(s.rep_type)


class _monad_manager:
    '''
    A mixin to help track monads as they are needed by statements that support them.
    '''

    monad_index: int = 0

    @classmethod
    def new_monad_ref(cls: _monad_manager):
        '''
        Return a new, unique, string that can be used as a monad reference
        '''
        assert cls.monad_index < 1000
        v = f'<mr-{cls.monad_index:03d}>'
        cls.monad_index += 1
        return v

    def __init__(self):
        self._monads: List[Tuple[str, str]] = []
        self._previous_statement_monad = False
        self._monad_ref: List[str] = []

    def add_monad(self, var_name: str, monad: str) -> int:
        '''
        Track a new monad

        Arguments:
            var_name        Name used in the monad (will be replaced)
            monad           String of the monad itself

        Returns:
            index           Index where this monad can be found in the resulting tuple.
                            Index 0 is always the main statement here, 1 will be the first
                            monad along for the ride, etc.
        '''
        # Have we added it before? If so, return that one.
        for i, m in enumerate(self._monads):
            if monad.replace(var_name, m[0]) == m[1]:
                return i + 1

        self._monads.append((var_name, monad))
        return len(self._monads)

    def render(self, var_name: str, main_func: str):
        '''Carry along the monads'''
        var_name_replacement = var_name
        if self._previous_statement_monad:
            var_name_replacement = _index_text_tuple(var_name, 0)
            main_func = main_func.replace(var_name, var_name_replacement)

            for mr in self._monad_ref:
                monad_references = re.findall(f'{mr}\\[[0-9]+\\]', main_func)
                for m in monad_references:
                    index_match = re.findall('\\[([0-9]+)\\]', m)
                    assert len(index_match) == 1
                    index = int(index_match[0])
                    replace_string = _index_text_tuple(var_name, index)
                    main_func = main_func.replace(m, replace_string)

        if len(self._monads) == 0:
            return main_func

        re_based = [m_func.replace(m_var, var_name) for m_var, m_func in self._monads]

        interior = ', '.join([main_func] + re_based)

        return f'({interior})'

    def carry_monad_forward(self, index: int) -> int:
        '''
        The previous statement is a tuple, containing a monad. We want to pass it down a statement.

        Arguments:
            index           Index in the previous tuple that we want to foward

        Returns:
            index           Index in this statement where this monad can be found
        '''
        assert index != 0, 'Internal error - never should pass main sequence through as monad'
        # Mark ourselves as a monad, man!
        self.prev_statement_is_monad()

        tv = new_var_name()
        return self.add_monad(tv, f'{tv}[{index}]')

    def prev_statement_is_monad(self):
        '''
        If the previous statement is a monad, then we will make sure the base access occurs with
        an index of zero.
        '''
        self._previous_statement_monad = True

    def set_monad_ref(self, monad_subst_string: str):
        '''
        The statements may contain monads - substitute their references
        '''
        self._monad_ref.append(monad_subst_string)

    def has_monads(self) -> bool:
        '''
        Return ture if there is a monad being carried along.
        '''
        return len(self._monads) > 0


class term_info:
    '''
    A term in an expression. Track all the info associated with it.
    '''
    def __init__(self, term: str, t: Type, monad_refs: List[str] = []):
        self.term = term
        self.type = t
        self.monad_refs = monad_refs


class statement_base:
    '''
    Base statement. Should never be created directly.
    '''
    def __init__(self, ast_rep: ast.AST, rep_type: Type):
        self._ast = ast_rep
        self.rep_type = rep_type

    def apply(self, stream: object) -> object:
        assert False, 'This should be overridden'

    # def apply_as_text(self, var_name: str) -> str:
    #     assert False, 'This should be overriden'

    def apply_as_function(self, stem: term_info) -> term_info:
        assert False, 'This should be overriden'

    def add_monad(self, var_name: str, monad: str) -> int:
        'Add a monad to be carried along'
        assert False, 'This should be overridden'

    def carry_monad_forward(self, index: int) -> int:
        'Carry a monad forward to this statement'
        assert False, 'This should be overridden'

    def has_monads(self) -> bool:
        assert False, 'this should be overridden'


class statement_unwrap_list(statement_base):
    '''
    A placeholder statement. Used to unwrap a type
    '''
    def __init__(self, ast_rep: ast.AST, rep_type: Type):
        assert _is_list(rep_type)
        statement_base.__init__(self, ast_rep, _unwrap_list(rep_type))


class statement_df(statement_base):
    '''
    Represents the dataframe/EventDataSet that is the source of all data.
    '''
    def __init__(self, ast_rep: ast.AST):
        statement_base.__init__(self, ast_rep, object)

    def apply(self, stream: object) -> object:
        from dataframe_expressions import ast_DataFrame
        from .hep_table import xaod_table
        assert isinstance(self._ast, ast_DataFrame)
        df = self._ast.dataframe
        assert isinstance(df, xaod_table)
        return df.event_source

    def add_monad(self, var_name: str, monad: str) -> int:
        '''
        No monad can be added on to this. Move quietly along. If someone calls apply then we
        will actually shut everything down.
        '''
        assert False, 'this should never be called'

    def __str__(self):
        return "EventSource"


def _render_as_function(s: Union[statement_select, statement_where],
                        var_name: term_info, operation: str) -> term_info:
    '''
    Helper function to render as a inline function
    '''
    # Pass all monad referces forward, we do not resolve them.
    monad_refs = s._monad_ref
    s._monad_ref = []
    if s._act_on_sequence:
        inner_var = new_var_name()
        inner_expr = s._func.term.replace(s._iterator.term, inner_var)
        expr = s.render(var_name.term,
                        f'{var_name.term}.{operation}(lambda {inner_var}: {inner_expr})')
        return term_info(expr, s.rep_type, monad_refs + var_name.monad_refs)
    else:
        return term_info(s.render(var_name.term,
                         s._func.term.replace(s._iterator.term, var_name.term)),
                         s.rep_type,
                         monad_refs)


class statement_select(_monad_manager, statement_base):
    '''
    Represents a transformation or mapping. Two types are handled:

        - Object transformation: df -> df.Select(lambda e: e.jets())
        - Sequence of objects tranformation:
            df -> df.Select(lambda e1: e1.Select(lambda e2: e2.jets()))
    '''
    def __init__(self, ast_rep: ast.AST, rep_type: Type, iterator: term_info,
                 function_text: term_info, is_sequence_of_objects: bool):
        '''
        Creates a select statement.
        '''
        statement_base.__init__(self, ast_rep, rep_type)
        _monad_manager.__init__(self)
        self._iterator = iterator
        self._func = function_text
        self._act_on_sequence = is_sequence_of_objects

    def _inner_lambda_text(self) -> str:
        '''
        Helper method
        '''
        if self._act_on_sequence:
            outter_var_name = new_var_name()
            inner_func = self.render(outter_var_name, f'{outter_var_name}'
                                     f'.Select(lambda {self._iterator.term}: {self._func.term})')
            lambda_text = f'lambda {outter_var_name}: {inner_func}'
        else:
            inner_func = self.render(self._iterator.term, self._func.term)
            lambda_text = f'lambda {self._iterator.term}: {inner_func}'
        return lambda_text

    def apply(self, seq: object) -> ObjectStream:
        # Build the lambda
        assert isinstance(seq, ObjectStream), 'Internal error'
        return seq.Select(self._inner_lambda_text())

    def __str__(self):
        return f'  .Select({self._inner_lambda_text()})'

    def apply_as_function(self, var_name: term_info) -> term_info:
        return _render_as_function(self, var_name, 'Select')


class statement_where(_monad_manager, statement_base):
    '''
    Represents a filtering. Two forms are handled.
        - Object filter: df -> df.Where(lambda e: e.jets())
        - Sequence of objects filter:
            df -> df.Select(lambda e1: e1.Where(lambda e2: e2.jets()))
    '''
    def __init__(self, ast_rep: ast.AST, rep_type: Type, iterator: term_info,
                 function_term: term_info, is_sequence_of_objects: bool):
        statement_base.__init__(self, ast_rep, rep_type)
        _monad_manager.__init__(self)
        self._iterator = iterator
        self._func = function_term
        self._act_on_sequence = is_sequence_of_objects

        for t in self._func.monad_refs:
            self.set_monad_ref(t)
            self.prev_statement_is_monad()

    def _inner_lambda_text(self) -> str:
        if self._act_on_sequence:
            outter_var_name = new_var_name()
            full_where_tuple = self.render(
                outter_var_name, f'{outter_var_name}.Where(lambda {self._iterator.term}: '
                f'{self._func.term})')
            lambda_text = f'lambda {outter_var_name}: {full_where_tuple}'
            return lambda_text
        else:
            lambda_text = f'lambda {self._iterator.term}: ' \
                f'{self.render(self._iterator.term, self._func.term)}'
            return lambda_text

    def apply(self, seq: object) -> ObjectStream:
        # Build the lambda
        assert isinstance(seq, ObjectStream), 'Internal error'
        if self._act_on_sequence:
            return seq.Select(self._inner_lambda_text())
        else:
            return seq.Where(self._inner_lambda_text())

    def apply_as_function(self, var_name: term_info) -> term_info:
        assert self._act_on_sequence, 'Not sure how Where works on single object'
        return _render_as_function(self, var_name, 'Where')

    def __str__(self):
        if self._act_on_sequence:
            return f'  .Select({self._inner_lambda_text()})'
        else:
            return f'  .Where({self._inner_lambda_text()})'


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
