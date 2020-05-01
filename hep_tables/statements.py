from __future__ import annotations

import ast
import re
from typing import List, Tuple, Type, cast

from func_adl import ObjectStream

from hep_tables.utils import _index_text_tuple, new_var_name, _unwrap_list, \
    _type_replace, new_term, _is_of_type, _is_list


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
        # TERMS already hold monads, should statements as well???

    def copy_monad_info(self, source: _monad_manager):
        'Copy over monad info from one to the other, will erase our initial info'
        self._monads = source._monads
        self._previous_statement_monad = source._previous_statement_monad
        self._monad_ref = source._monad_ref
        return self

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
            index           Index in the previous tuple that we want to forward

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
        if monad_subst_string not in self._monad_ref:
            self._monad_ref.append(monad_subst_string)

    def has_monads(self) -> bool:
        '''
        Return true if there is a monad being carried along.
        '''
        return len(self._monads) > 0

    def has_monad_refs(self) -> bool:
        return len(self._monad_ref) > 0


class term_info:
    '''
    A term in an expression. Track all the info associated with it.
    '''
    def __init__(self, term: str, t: Type, monad_refs: List[str] = []):
        self.term = term
        self.type = t
        self.monad_refs = monad_refs

    def __str__(self):
        return f'{self.term}: {self.type}'.replace("typing.", "")

    def __repr__(self):
        return f'{self.term}: {self.type}'.replace("typing.", "")

    def has_monads(self) -> bool:
        return len(self.monad_refs) > 0


class statement_base:
    '''
    Base statement. Should never be created directly.
    '''
    def __init__(self, ast_rep: ast.AST, input_sequence_type: Type,
                 result_sequence_type: Type):
        self._ast = ast_rep
        self._input_sequence_type = input_sequence_type
        self._result_sequence_type = result_sequence_type

    @property
    def result_type(self) -> Type:
        return self._result_sequence_type

    def apply(self, stream: object) -> object:
        assert False, 'This should be overridden'

    def apply_as_function(self, stem: term_info) -> term_info:
        assert False, 'This should be overridden'

    def add_monad(self, var_name: str, monad: str) -> int:
        'Add a monad to be carried along'
        assert False, 'This should be overridden'

    def carry_monad_forward(self, index: int) -> int:
        'Carry a monad forward to this statement'
        assert False, 'This should be overridden'

    def has_monads(self) -> bool:
        assert False, 'this should be overridden'

    def unwrap(self) -> statement_base:
        '''
        If we operate on a list, then produce a statement that operates on whatever
        is inside that list. If we were List[object] -> List[bool], and our iterator was
        object, then return the same statement but that works object -> bool.
        '''
        assert False, 'This should be overridden'

    def wrap(self) -> statement_base:
        '''
        If we operate on a list, then produce a statement that operates on whatever
        is inside that list. If we were object -> bool, and our iterator was
        object, then return the same statement but that works List[object] -> List[bool].
        '''
        assert False, 'This should be overridden'

    def unwrap_if_possible(self) -> statement_base:
        if _is_list(self._result_sequence_type) and _is_list(self._input_sequence_type):
            return self.unwrap()

        if (not _is_list(self._result_sequence_type)) \
                and (not _is_list(self._input_sequence_type)):
            return self

        r_type = _unwrap_list(self._result_sequence_type) if _is_list(self._result_sequence_type) \
            else self._result_sequence_type

        return statement_base(self._ast, self._input_sequence_type, r_type)

    def set_monad_ref(self, monad_subst_string: str):
        assert False, 'Must be overridden'

    def prev_statement_is_monad(self):
        assert False, 'Must be overridden'


class statement_df(statement_base):
    '''
    Represents the dataframe/EventDataSet that is the source of all data.
    We have no input type, and our result is an "event" object.
    '''
    def __init__(self, ast_rep: ast.AST):
        statement_base.__init__(self, ast_rep, object, object)

    def apply(self, stream: object) -> object:
        # Note that stream is ignored here.
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


class statement_base_iterator(_monad_manager, statement_base):
    '''
    Base class for statements that have iterators
    '''
    def __init__(self, ast_rep: ast.AST, input_sequence_type: Type,
                 result_sequence_type: Type, iterator: term_info,
                 function: Type, pass_through: bool):
        '''
        Arguments:
            pass_through        Input and output types are the same, function result
                                is not checked.
        '''
        statement_base.__init__(self, ast_rep, input_sequence_type,
                                result_sequence_type)
        _monad_manager.__init__(self)
        self._iterator = iterator
        self._func = function
        for m in function.monad_refs:
            self.set_monad_ref(m)

        # Check that the types make sense. The earlier we catch this
        # the easier it is to debug.
        final_type = _type_replace(input_sequence_type,
                                   iterator.type,
                                   function.type)
        assert final_type is not None, \
            f'Internal error - cannot find iterator type {iterator.type} ' \
            f'in sequence type {str(input_sequence_type)}'

        assert pass_through or _is_of_type(final_type, result_sequence_type), \
            'Internal error: types not consistent in iterator statement: ' \
            f'input: {input_sequence_type}, result: {result_sequence_type}, ' \
            f'iterator: {iterator.type}, function: {function.type}'

    def _inner_lambda(self, iter: term_info, op: str) -> term_info:
        '''
        Helper method to render the inner lambda text, which is
        the same in both ObjectStream and text stream cases.
        '''
        # Return a properly nested function!
        return self._render_inner(self._input_sequence_type, iter, op)

    def _render_as_function(self, sequence: term_info, op: str,
                            render_monads: bool = False) -> term_info:
        '''
        Helper function to render as a inline function ready to use in code.
        '''
        assert _is_of_type(self._input_sequence_type, sequence.type), \
            f'Internal Error: sequence type {self._input_sequence_type} not compatible ' \
            f'with iterator sequence type {sequence.type}.'

        # Pass all monad references forward, we do not resolve them.
        monad_refs = self._monad_ref
        if not render_monads:
            self._monad_ref = []

        # Next, we have to code up the outter statement
        inner_expr = self._inner_lambda(sequence, op)
        inner_expr_txt = self.render(sequence.term, inner_expr.term)
        self._monad_ref = monad_refs
        return term_info(inner_expr_txt, inner_expr.type, monad_refs + sequence.monad_refs)

    def _render_inner(self, in_type: Type, iter: term_info, op: str) -> term_info:
        '''
        Recursively nest the statement as needed.
        '''
        if _is_of_type(in_type, self._iterator.type):
            inner_func = self._func.term
            inner_func = inner_func.replace(self._iterator.term, iter.term)
            return term_info(inner_func, self._func.type)
        else:
            v = new_term(_unwrap_list(in_type))
            unwrapped = _unwrap_list(in_type)
            inner_func = self._render_inner(unwrapped, v, op)
            use_op = op if _is_of_type(unwrapped, self._iterator.type) else 'Select'
            inner_type = inner_func.type if use_op == 'Select' else unwrapped
            return term_info(f'{iter.term}.{use_op}(lambda {v.term}: {inner_func.term})',
                             List[inner_type])

    def clone_with_types(self, type_input: Type, type_output: Type) -> statement_base_iterator:
        assert False, 'This needs to be overridden'

    def unwrap(self) -> statement_base:
        assert _is_list(self._result_sequence_type), \
            f'Cannot unwrap list of type {self._result_sequence_type}'

        new_input_type = _unwrap_list(self._input_sequence_type)
        new_result_type = _unwrap_list(self._result_sequence_type)

        return cast(statement_base, self.clone_with_types(new_input_type, new_result_type)
                    .copy_monad_info(self))

    def wrap(self) -> statement_base:
        new_input_type = List[self._input_sequence_type]
        new_result_type = List[self._result_sequence_type]

        return cast(statement_base, self.clone_with_types(new_input_type, new_result_type)
                    .copy_monad_info(self))


class statement_select(statement_base_iterator):
    '''
    Represents a transformation or mapping. Two types are handled:

        - Object transformation: df -> df.Select(lambda e: e.jets())
        - Sequence of objects tranformation:
            df -> df.Select(lambda e1: e1.Select(lambda e2: e2.jets()))
    '''
    def __init__(self, ast_rep: ast.AST, input_sequence_type: Type,
                 result_sequence_type, iterator: term_info,
                 function: term_info):
        '''
        Creates a select statement.
        '''
        statement_base_iterator.__init__(self, ast_rep, input_sequence_type,
                                         result_sequence_type, iterator,
                                         function, False)

    def clone_with_types(self, type_input: Type, type_output: Type) -> statement_base_iterator:
        return statement_select(self._ast, type_input, type_output,
                                self._iterator, self._func)

    def apply(self, seq: object) -> ObjectStream:
        assert isinstance(seq, ObjectStream), 'Internal error'
        inner_lambda = self._render_as_function(term_info(self._iterator.term,
                                                          self._input_sequence_type),
                                                'Select', True)
        return seq.Select(f'lambda {self._iterator.term}: {inner_lambda.term}')

    def __str__(self):
        inner_lambda = self._render_as_function(term_info(self._iterator.term,
                                                          self._input_sequence_type),
                                                'Select', True)
        return f'  .Select(lambda {self._iterator.term}: {inner_lambda.term})'

    def apply_as_function(self, sequence: term_info) -> term_info:
        return self._render_as_function(sequence, 'Select')


class statement_where(statement_base_iterator):
    '''
    Represents a filtering. Two forms are handled.
        - Object filter: df -> df.Where(lambda e: e.jets())
        - Sequence of objects filter:
            df -> df.Select(lambda e1: e1.Where(lambda e2: e2.jets()))
    '''
    def __init__(self, ast_rep: ast.AST, input_sequence_type: Type,
                 iterator: term_info,
                 function: term_info):

        # Get some object invariants setup right
        assert function.type == bool, f'Where function ({function.term}) must be type <bool>, not <{function.type.__name__}>'

        statement_base_iterator.__init__(self, ast_rep, input_sequence_type,
                                         input_sequence_type, iterator,
                                         function, True)

        # TODO: does this belong here or in the select statement?
        for t in self._func.monad_refs:
            self.set_monad_ref(t)
            self.prev_statement_is_monad()

    def clone_with_types(self, type_input: Type, type_output: Type) -> statement_base_iterator:
        return statement_where(self._ast, type_input,
                               self._iterator, self._func)

    def apply(self, seq: object) -> ObjectStream:
        assert isinstance(seq, ObjectStream), 'Internal error'
        inner_lambda = self._render_as_function(term_info(self._iterator.term,
                                                          self._input_sequence_type),
                                                'Where', True)
        if 'Where' in inner_lambda.term:
            return seq.Select(f'lambda {self._iterator.term}: {inner_lambda.term}')
        else:
            return seq.Where(f'lambda {self._iterator.term}: {inner_lambda.term}')

    def apply_as_function(self, var_name: term_info) -> term_info:
        return self._render_as_function(var_name, 'Where')

    def __str__(self):
        inner_lambda = self._render_as_function(term_info(self._iterator.term,
                                                          self._input_sequence_type),
                                                'Where', True)
        if 'Where' in inner_lambda.term:
            return f'  .Select(lambda {self._iterator.term}: {inner_lambda.term})'
        else:
            return f'  .Where(lambda {self._iterator.term}: {inner_lambda.term})'


class statement_constant(statement_base):
    '''
    A bit of a weird one - returns  a constant that should be used
    directly as input for the next thing.
    '''
    def __init__(self, ast_rep: ast.AST, value: object, rep_type: Type):
        statement_base.__init__(self, ast_rep, object, rep_type)
        self._value = value

    def apply(self, stream: object) -> object:
        return self._value

    def apply_as_function(self, stem: term_info) -> term_info:
        return term_info(str(self._value), self._result_sequence_type)
