from __future__ import annotations
import ast
from typing import Dict, List, Optional, Tuple, Type

from dataframe_expressions import (
    ast_Callable, ast_DataFrame, ast_Filter, ast_FunctionPlaceholder,
    render_callable, render_context)
from func_adl_xAOD import use_exe_servicex

from .statements import (
    statement_base, statement_constant, statement_df, statement_select,
    statement_unwrap_list, statement_where)
from .utils import (
    _find_root_expr, new_var_name, to_args_from_keywords)


class _ast_VarRef(ast.AST):
    'An internal AST when we want to replace an ast with a variable reference inline'
    def __init__(self, name: str, t: Type):
        ast.AST.__init__(self)
        self.name = name
        self.type = t
        self._fields = ('name', 'type')


class replace_an_ast:
    '''
    For use in with clauses. Do not create directly!
    '''
    def __init__(self, tracker: _statement_tracker, source: ast.AST, dest: ast.AST):
        self._tracker = tracker
        self._source = source
        self._dest = dest
        self._done = False

    def __enter__(self):
        # TODO: WARNING - this might be hiding aliases, if we ever want to do two things
        # of the same object, then this might cause us problems!!!
        if self._tracker.lookup_ast(self._source) is None:
            self._tracker.ast_replacements.append((self._source, self._dest))
            self._done = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._done:
            r = self._tracker.ast_replacements.pop()
            assert r[0] is self._source


class _statement_tracker:
    '''
    Track statements in seperate stacks. We have a parent link so we can
    look all the way back in the stack if need be.
    '''
    def __init__(self, start_sequence: statement_base, parent: Optional[_statement_tracker]):
        self._parent_tracker = parent
        self.statements: List[statement_base] = []
        self.sequence = start_sequence
        self.base_sequence = start_sequence
        self.ast_replacements: List[Tuple[ast.AST, ast.AST]] = []

    def carry_monad_forward(self, a: ast.AST) -> int:
        '''
        Go back in our sequence of statements to find the sequence `a`. Once found, bring it
        forward as a monad for later use.
        '''
        index = 0 if self.base_sequence._ast is a else None
        if index is None:
            possible = [(i, s) for i, s in enumerate(self.statements) if s._ast is a]
            assert len(possible) != 0 or self._parent_tracker is not None, \
                f'Internal error, unable capture {ast.dump(a)}'

            if len(possible) > 0:
                index = possible[0][0] + 1

        if index is None:
            # last chance - someone above us?
            m_index = self._parent_tracker.carry_monad_forward(a)
            index = -1
        else:
            m_name = new_var_name()
            m_index = self.statements[index].add_monad(m_name, m_name)

        # Now bring the monad as far forward as we can.
        for s in self.statements[index + 1:]:
            m_index = s.carry_monad_forward(m_index)

        return m_index

    def substitute_ast(self, source_ast: ast.AST, replace_with: ast.AST) -> replace_an_ast:
        '''
        Add to the replacement stack. And do it in order.
        '''
        return replace_an_ast(self, source_ast, replace_with)

    def lookup_ast(self, a: ast.AST) -> Optional[ast.AST]:
        '''
        See if we can find an ast for replacement, return it or None
        '''
        for rp in reversed(self.ast_replacements):
            if rp[0] is a:
                return rp[1]

        return None if self._parent_tracker is None else self._parent_tracker.lookup_ast(a)


class _map_to_data(_statement_tracker, ast.NodeVisitor):
    '''
    The main driver of the translation. We build up a LINQ query - this means we have a
    "sequence" that we are manipulating as we move through by either transforming (with
    Select) or filtering (with Where). As such, we maintain something that tells us what
    the current sequence is.
    '''
    def __init__(self, base_sequence: statement_base, context: render_context,
                 p_tracker: _statement_tracker):
        '''
        Create a mapper

        Arguments:
            base_sequence           This is the sequence infomration that we start wit.
        '''
        ast.NodeVisitor.__init__(self)
        _statement_tracker.__init__(self, base_sequence, p_tracker)
        self.context = context

    def visit(self, a: ast.AST):
        '''
        Check to see if the ast we are about to visit is the one that represents the current
        stream. If that is the case, then there is no need to generate any further code here.
        '''
        if a is self.sequence._ast:
            return

        # Is this on our substitution list?
        sub = self.lookup_ast(a)
        if sub is not None:
            self.visit(sub)
            return

        ast.NodeVisitor.visit(self, a)

    def visit_ast_DataFrame(self, a: ast_DataFrame):
        assert False, 'We should never get this low in the chain'

    def visit_ast_Filter(self, a: ast_Filter):
        'Get the expression and apply the filter'
        self.visit(a.expr)

        # How we do the filter depends on what we are looking at
        # 1. object (implied sequence, one item per event):
        #       d.Where(lambda e: e > 10)
        # 2. List[object] (explicit sequence, one list per event):
        #       d.Select(lambda e: e.Where(labmda ep: ep > 10))
        if self.sequence.rep_type is List[object]:
            # Since this guy is a sequence, we have to turn it into not-a sequence for processing.
            filter_sequence, trm = _render_expression(
                statement_unwrap_list(self.sequence._ast, self.sequence.rep_type), a.filter,
                self.context, self)
            act_on_sequence = True
            assert trm.term == 'main_sequence', 'Unexpected term type for filter expression'
        else:
            filter_sequence, trm = _render_expression(self.sequence, a.filter, self.context, self)
            act_on_sequence = False
            assert trm.term == 'main_sequence', 'Unexpected term type for filter expression'

        assert len(filter_sequence) > 0
        var_name = new_var_name()
        stem = var_name
        for s in filter_sequence:
            stem = s.apply_as_function(stem)
        st = statement_where(a, self.sequence.rep_type,
                             var_name, stem,
                             act_on_sequence)
        self.statements.append(st)
        self.sequence = st

    def visit_Attribute(self, a: ast.Attribute):
        # Get the stream up to the base expressoin.
        # if a.value is not self.sequence._ast:
        statements, term = _render_expression(self.sequence, a.value, self.context, self)

        if term.term == 'main_sequence':
            if len(statements) > 0:
                self.statements += statements
                self.sequence = statements[-1]
        else:
            assert len(statements) == 0, \
                'Internal programming error - cannot deal with statements and term'
            vn = new_var_name()
            st_select = statement_select(a.value, term.type, vn,
                                         term.term, term.type is List[object])
            self.statements.append(st_select)
            self.sequence = st_select

        # Now we need to "select" a level here. This means doing a call.
        name = a.attr
        self.append_call(a, name, None)

    def render_locally(self, v: ast.AST):
        '''
        Render v locally as data
        '''
        from .utils import _find_dataframes
        from .local import default_col_name
        from func_adl import ObjectStream

        base_ast_df = _find_dataframes(v)

        # TODO: THis is the same code as in make_local - perhaps????
        mapper = _map_to_data(statement_df(base_ast_df), self.context, self)
        mapper.visit(v)

        result = base_ast_df.dataframe.event_source
        for seq in mapper.statements:
            result = seq.apply(result)

        if isinstance(result, ObjectStream):
            return result.AsAwkwardArray(['col1']).value(use_exe_servicex)[default_col_name]
        else:
            return result

    def visit_call_histogram(self, value: ast.AST, args: List[ast.AST],
                             keywords: List[ast.keyword], a: ast.Call):
        '''
        Generate a histogram. We can't do this in the abstract, unfortunately,
        as we have to get the data and apply the numpy.histogram at this upper level.
        '''
        assert len(args) == 0, 'Do not know how to process extra args for numpy.histogram'
        data = self.render_locally(value)
        if hasattr(data, 'flatten'):
            data = getattr(data, 'flatten')()
        import numpy as np
        kwargs = to_args_from_keywords(keywords)
        result = np.histogram(data, bins=kwargs['bins'],
                              range=kwargs['range'],
                              density=kwargs['density'])
        st = statement_constant(a, result, type(result))
        self.statements.append(st)
        self.sequence = st

    def visit_call_map(self, value: ast.AST, args: List[ast.AST],
                       keywords: List[ast.keyword], a: ast.Call):
        '''
        We are going to map a lambda function onto this sequence.
        '''
        assert len(args) == 1
        callable = args[0]
        assert isinstance(callable, ast_Callable)

        # We have to render the callable at this point.
        self.visit(value)
        expr = render_callable(callable, self.context, callable.dataframe)

        # In that expr there may be captured variables, or references to things that
        # are not in `value`. If that is the case, that means we need to add a monad to fetch
        # them from earlier in the process.
        root_expr = _find_root_expr(expr, self.sequence._ast)
        if root_expr is self.sequence._ast:
            self.visit(expr)
        else:
            monad_index = self.carry_monad_forward(root_expr)

            with self.substitute_ast(
                    root_expr, _ast_VarRef(f'<monad-ref>[{monad_index}]', object)):
                select_var = new_var_name()
                with self.substitute_ast(
                        self.sequence._ast, _ast_VarRef(select_var, object)):
                    # expr = _ast_replace(expr, root_expr,
                    #                     _ast_VarRef(f'<monad-ref>[{monad_index}]', object))
                    # expr = _ast_replace(expr, self.sequence._ast, _ast_VarRef(select_var,
                    #                     object))

                    if self.sequence.rep_type is List[object]:
                        # Since this guy is a sequence, we have to turn it into not-a sequence for
                        # processing.
                        trm = _resolve_arg(statement_unwrap_list(self.sequence._ast,
                                                                 self.sequence.rep_type),
                                           expr, self.context, self)
                        # func_sequence, trm = _render_expression(
                        #     statement_unwrap_list(self.sequence._ast, self.sequence.rep_type),
                        #     expr,
                        #     self.context, self)
                        act_on_sequence = True
                    else:
                        assert False, 'This path has not been tested - DANGER'
                        # func_sequence, trm = _render_expression(self.sequence, expr,
                        # self.context, self)
                        # act_on_sequence = False
                    # assert trm.term == 'main_sequence' or len(func_sequence) == 0, \
                    #     'Unexpected term type for filter expression'

            # Build the select statement as an internal function.

            # if len(func_sequence) == 0:
            #     internal_func = trm.term
            # else:
            #     internal_func = select_var
            #     for s in func_sequence:
            #         internal_func = s.apply_as_function(internal_func)

            st = statement_select(a, List[object],
                                  select_var, trm.term,
                                  act_on_sequence)
            st.prev_statement_is_monad()
            self.statements.append(st)
            self.sequence = st

            # TODO: pull this stuff above out - it is common!

    def visit_Call(self, a: ast.Call):
        # Math function calls are treated like expressions
        if isinstance(a.func, ast.Attribute):
            if a.func.attr in _known_simple_math_functions:
                r, _ = _render_expression(self.sequence, a, self.context, self)
                self.statements = self.statements + r
            elif hasattr(self, f'visit_call_{a.func.attr}'):
                m = getattr(self, f'visit_call_{a.func.attr}')
                m(a.func.value, a.args, a.keywords, a)
            else:
                # We are now accessing a column or collection off a event or other collection.
                # The LINQ, functional, way of doing this is by going down a level.
                self.visit(a.func.value)

                # TODO: resovle arg should be a call to the expression thing!
                resolved_args = [_resolve_arg(self.sequence, arg, self.context, self)
                                 for arg in a.args]
                name = a.func.attr
                self.append_call(a, name, [r.term for r in resolved_args])
        elif isinstance(a.func, ast_FunctionPlaceholder):
            # We will embed this in a select statement. And the sequence items will
            # need to be explicitly referenced in the arguments.
            var_name = new_var_name()

            def do_resolve(arg):
                with self.substitute_ast(self.sequence._ast, _ast_VarRef(var_name, object)):
                    # arg1 = _ast_replace(arg, self.sequence._ast, _ast_VarRef(var_name, object))
                    return _resolve_arg(self.sequence, arg, self.context, self)

            resolved_args = [do_resolve(arg) for arg in a.args]
            name = a.func.callable.__name__
            for t in resolved_args:
                # We can't deal with arrays as arguments yet.
                assert t.type is not List[object], \
                    f'Functions with array arguments are not supported ({name})'
            args = ', '.join(t.term for t in resolved_args)
            st = statement_select(a, object, var_name,
                                  f'{name}({args})', self.sequence.rep_type is List[object])
            self.statements.append(st)
            self.sequence = st
            pass
        else:
            assert False, 'Function calls can only be method calls or place holders'

    def visit_BinOp(self, a: ast.BinOp):
        # TODO: Should support 1.0 / j.pt as well as j.pt / 1.0
        r, _ = _render_expression(self.sequence, a, self.context, self)
        self.statements = self.statements + r

    def append_call(self, a: ast.AST, name_of_method: str, args: Optional[List[str]]) -> None:
        'Append a call onto the call chain that will look at this method'
        # Build the function call
        arg_text = "" if args is None else ", ".join([str(ag) for ag in args])
        function_call = f'{name_of_method}({arg_text})'
        iterator_name = new_var_name()
        expr = f'{iterator_name}.{function_call}'

        # The result type of the sequence after we are done. Will depend on what we are currently
        # working on
        input_type, result_type = _type_system(name_of_method)
        working_on_sequence = self.sequence.rep_type is List[object]
        if working_on_sequence:
            if input_type is not List[object]:
                # Input is a single object, and we are applying it to a list.
                # Heuristics: we do a map operation.
                result_type = List[result_type]
            else:
                # Input is a List, thus we eat this guy as if it was a single
                # object. This might be something like Count or Sum
                working_on_sequence = False
        else:
            if input_type is List[object]:
                assert False, 'Do not know how to turn a single object into a list'

        # Finally, build the map statement, and then update the current sequence.
        select = statement_select(a, result_type, iterator_name, expr, working_on_sequence)
        self.statements.append(select)
        self.sequence = select


class term_info:
    def __init__(self, term: str, t: Type):
        self.term = term
        self.type = t


def _render_expression(current_sequence: statement_base, a: ast.AST,
                       context: render_context, p_tracker: Optional[_statement_tracker]) \
        -> Tuple[List[statement_base], term_info]:
    '''
    Render an expression. If the expression contains linq selections stuff, then we will
    have to go back and render those as well by making a call back into _map.

    Arguments:
        current_sequence        The current active sequence that we are looking at
        a                       The expression ast to be parsed
        context                 Rendering context - used if we have to re-run a lambda or similar

    Returns:
        statements              A list of statements to be appended or referenced from the current
                                sequence.
        term                    'main_sequence' or a string term representing the expression

    '''
    class render_expression(_statement_tracker, ast.NodeVisitor):
        def __init__(self, current_sequence: statement_base, context: render_context,
                     p_tracker: _statement_tracker):
            ast.NodeVisitor.__init__(self)
            _statement_tracker.__init__(self, current_sequence, p_tracker)
            self.term_stack: List[term_info] = []
            self.context = context

        def visit(self, a: ast.AST):
            # Is there a substititon?
            sub = self.lookup_ast(a)
            if sub is not None:
                self.visit(sub)
                return

            ast.NodeVisitor.visit(self, a)

        def binary_op_statement(self, operator: ast.AST, a_left: ast.AST, a_right: ast.AST):
            '''
            Create the statements needed for a binary operator
            '''
            s_left, left = _render_expression(self.sequence, a_left, self.context, self)
            s_right, right = _render_expression(self.sequence, a_right, self.context, self)

            assert (len(s_right) == 0 or s_right[-1].rep_type is not List[object]) \
                or (len(s_left) == 0 or s_left[-1].rep_type is not List[object])

            def do_statements(s: List[statement_base], t: term_info, var_name: str):
                if t.term != 'main_sequence':
                    return t.term
                if len(s) == 0:
                    return var_name
                if s[-1].rep_type is not List[object]:
                    stem = var_name
                    for single in s:
                        stem = single.apply_as_function(stem)
                    return stem
                else:
                    self.statements += s
                    self.sequence = s[-1]
                    return var_name

            var_name = new_var_name()
            l_l = do_statements(s_left, left, var_name)
            l_r = do_statements(s_right, right, var_name)

            # Is the compare between two "terms" we know, or a sequence?
            op = _known_operators[type(operator)]
            if left.term != 'main_sequence' and right.term != 'main_sequence':
                self.term_stack.append(term_info(f'({l_l} {op} {l_r})', left.type))
            else:
                l_l = l_l if l_l is not None else var_name
                l_r = l_r if l_r is not None else var_name
                expr = f'({l_l} {op} {l_r})'
                rep_type = self.sequence.rep_type
                self.statements.append(statement_select(a, bool, var_name, expr,
                                                        rep_type is List[object]))
                self.term_stack.append(term_info('main_sequence', List[bool]))

        def visit_Compare(self, a: ast.Compare):
            # Need better protection here - if visit doesn't render something
            # even if we are deep in an expression... This will catch internal errors
            # in a production system when someone ueses a "new" feature of python we
            # don't have yet. As it stands, it will still cause an assertion failure, it will
            # just potentially be far away from the place the problem actually occured.
            assert len(a.comparators) == 1
            assert len(a.ops) == 1
            if type(a.ops[0]) not in _known_operators:
                raise Exception(f'Unknown operator {str(a.ops[0])} - cannot translate.')

            self.binary_op_statement(a.ops[0], a.left, a.comparators[0])

        def visit_BinOp(self, a: ast.BinOp):
            '*, /, +, and -'
            self.binary_op_statement(a.op, a.left, a.right)

        def visit_BoolOp(self, a: ast.BoolOp):
            'and or'
            def process_term(r: render_expression, a: ast.AST):
                s_left, left = _render_expression(r.sequence, a, self.context, self)
                # r.statements += s_left
                return s_left, left

            assert len(a.values) == 2
            terms = [process_term(self, a) for a in a.values]
            assert [t[1].term for t in terms].count('main_sequence') == len(terms)

            var_name = new_var_name()
            l_stem = var_name
            for s in terms[0][0]:
                l_stem = s.apply_as_function(l_stem)

            r_stem = var_name
            for s in terms[1][0]:
                r_stem = s.apply_as_function(r_stem)

            expr = f'({l_stem}) {_known_operators[type(a.op)]} ({r_stem})'
            st = statement_select(a, bool, var_name, expr, False)
            self.statements.append(st)
            self.sequence = st
            self.term_stack.append(term_info('main_sequence', List[object]))

        def visit_Num(self, a: ast.Num):
            'A number term should be pushed into the stack'
            self.term_stack.append(term_info(str(a.n), 'float'))

        def visit_Str(self, a: ast.Str):
            'A string should be pushed onto the stack'
            self.term_stack.append(term_info(f'"{a.s}"', str))

        def visit__ast_VarRef(self, a: _ast_VarRef):
            self.term_stack.append(term_info(a.name, a.type))

        def process_with_mapper(self, a: ast.AST):
            '''
            Use the main reducer to parse this ast.
            '''
            mapper = _map_to_data(self.sequence, self.context, self)
            mapper.visit(a)
            if len(mapper.statements) > 0:
                self.statements = self.statements + mapper.statements
                self.sequence = self.statements[-1]

            # The stream is now the term we want to use. In order to do this we'll now have
            # to deal with a sequence reference. This is the main sequence, so we want to leave
            # that as the term.
            self.term_stack.append(term_info("main_sequence", List[object]))

        def visit_Attribute(self, a: ast.Attribute):
            '''
            Attributes are calls or references - so we need to use the other reducer to
            understand them
            '''
            self.process_with_mapper(a)

        def visit_ast_Filter(self, a: ast_Filter):
            '''
            Filters need the full processing power
            '''
            self.process_with_mapper(a)

        def visit_Call(self, a: ast.Call):
            '''
            Deal with math functions.
            '''
            assert isinstance(a.func, ast.Attribute) or isinstance(a.func, ast_FunctionPlaceholder)

            if isinstance(a.func, ast.Attribute):
                if a.func.attr not in _known_simple_math_functions:
                    self.process_with_mapper(a)
                else:
                    assert len(a.args) == 0

                    self.visit(a.func.value)
                    v = self.term_stack.pop()

                    if v.term != 'main_sequence':
                        self.term_stack.append(
                            term_info(f'{_known_simple_math_functions[a.func.attr]}({v.term})',
                                      v.type))
                    else:
                        var_name = new_var_name()
                        expr = f'({_known_simple_math_functions[a.func.attr]}({var_name}))'
                        self.statements.append(
                            statement_select(a, object, var_name, expr,
                                             self.sequence.rep_type is List[object]))
                        self.term_stack.append(term_info('main_sequence', List[float]))
            else:
                # We are going to need to grab each argument (and in some cases we will
                # need to use monads to track what the different arguments are going to
                # be using)
                # We need this to be "clean" because we can't tell from context what should
                # be inline and what should be outter.
                self.process_with_mapper(a)

    r = render_expression(current_sequence, context, p_tracker)
    r.visit(a)
    assert len(r.term_stack) == 1 or (len(r.term_stack) == 0 and len(r.statements) == 0)
    return r.statements, \
        r.term_stack[0] if len(r.term_stack) != 0 else term_info('main_sequence', List[object])


# Known operators and their "text" rep.
_known_operators: Dict[Type, str] = {
    ast.Div: '/',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Add: '+',
    ast.Gt: '>',
    ast.GtE: '>=',
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.Eq: '==',
    ast.NotEq: '!=',
    ast.And: 'and',
    }


def _resolve_arg(curret_sequence: statement_base, expr: ast.AST, context: render_context,
                 p_tracker: _statement_tracker) \
        -> term_info:
    '''
    Resovel an expression in-line.

    Arguments:
        current_sequence        The current sequence is what - we can grab this from outside
        expr                    Expression representing the argument
        context                 The render context to pass through in case rendering is needed.
    '''
    # TODO: this should be resolve_inline_expression, and should be used in several places
    # in the above code. REFACTOR!!!
    # How we do the filter depends on what we are looking at
    # 1. object (implied sequence, one item per event):
    #       d.Where(lambda e: e > 10)
    # 2. List[object] (explicit sequence, one list per event):
    #       d.Select(lambda e: e.Where(labmda ep: ep > 10))
    if curret_sequence.rep_type is List[object]:
        # Since this guy is a sequence, we have to turn it into not-a sequence for processing.
        filter_sequence, trm = _render_expression(
            statement_unwrap_list(curret_sequence._ast, curret_sequence.rep_type), expr,
            context, p_tracker)
        # act_on_sequence = True
    else:
        filter_sequence, trm = _render_expression(curret_sequence, expr, context, p_tracker)
        # act_on_sequence = False

    assert (trm.term != 'main_sequence' and len(filter_sequence) == 0) \
        or (trm.term == 'main_sequence' and len(filter_sequence) > 0)

    if len(filter_sequence) > 0:
        # If we have to create a new variable here, then probably somethign has gone wrong.
        a_resolved = p_tracker.lookup_ast(curret_sequence._ast)
        assert (a_resolved is not None) \
            or (filter_sequence[0].apply_as_function('bogus').find('bogus') < 0)
        assert a_resolved is None or isinstance(a_resolved, _ast_VarRef)
        stem = 'bogus' if a_resolved is None else a_resolved.name
        for s in filter_sequence:
            stem = s.apply_as_function(stem)
        return term_info(stem, filter_sequence[-1].rep_type)
    else:
        return trm


_known_types = {
    'jets': (object, List[object]),
    'Jets': (object, List[object]),
    'Electrons': (object, List[object]),
    'Count': (List[object], object)
}


def _type_system(n: str) -> Tuple[Type, Type]:
    '''
    Determine the type of method/prop that is being accessed. This
    is using heuristics.
    TODO: This needs to be robust! Big can of worms

    Args:
        n           Name of the method that we are looking at. No context is given, just name.

    Returns:
        arg_type    What does it operate on (list or object)?
        rtn_type    What does it return
    '''
    return _known_types[n] if n in _known_types else (object, object)


# List of math functions we translate into something similar in the LINQ code.
_known_simple_math_functions = {
    'abs': 'abs',
    # numpy functions
    'absolute': 'abs',
}
