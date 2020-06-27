from __future__ import annotations
import ast
import inspect
from typing import Callable, Dict, List, Optional, Tuple, Type, cast
from contextlib import ExitStack


from dataframe_expressions import (
    ast_Callable, ast_DataFrame, ast_Filter, ast_FunctionPlaceholder,
    render_callable, render_context)

from .statements import (
    _monad_manager, statement_base, statement_constant, statement_df,
    statement_select, statement_where, term_info)
from .utils import (
    _find_root_expr, _is_list, _type_replace, _unwrap_list, _unwrap_if_possible,
    new_var_name, new_term, to_args_from_keywords)


class RenderException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)


def curry(f: Callable) -> Callable:
    '''
    This will take the given function `f` and return a new function that needs only the
    last argument.

    @curry
    def test(a, b, c):
        a+b+c

    then you can do:
        t1 = test(a, b)
        assert t1(c) == a+b+c
    '''
    nargs = len(inspect.signature(f).parameters)

    def build_curried_function(*args) -> Callable:
        def apply_last_arg(last_arg):
            new_args = args + (last_arg,)
            return f(*new_args)

        assert len(args) == (nargs - 1)
        return apply_last_arg

    return build_curried_function


class _ast_VarRef(ast.AST):
    'An internal AST when we want to replace an ast with a variable reference inline'
    _fields = ('term',)

    def __init__(self, v: term_info):
        ast.AST.__init__(self)
        self.term = v

    def __str__(self):
        return f'{self.term.term}: {self.term.type}'

    def __repl__(self):
        return f'{self.term.term}: {self.term.type}'


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


def _render_expresion_as_transform(tracker: _statement_tracker, context: render_context,
                                   a: ast.AST):
    '''
    Render an expression and add into the sequence. If this turns out to be a term,
    then as a select.
    '''
    statements, term = _render_expression(tracker.sequence, a, context, tracker)

    if term.term == 'main_sequence':
        if len(statements) > 0:
            tracker.statements += statements
            tracker.sequence = statements[-1]
    else:
        assert len(statements) == 0, \
            f'Internal programming error - cannot deal with statements and term {term.term}'
        vn = new_term(_unwrap_list(tracker.sequence.result_type)
                      if _is_list(tracker.sequence.result_type)
                      else tracker.sequence.result_type)
        in_type = tracker.sequence.result_type
        out_type = _type_replace(in_type, vn.type, term.type)
        st_select = statement_select(a, in_type, out_type, vn, term)
        tracker.statements.append(st_select)
        tracker.sequence = st_select


def _render_callable(a: ast.AST, callable: ast_Callable, context: render_context,
                     tracker: _statement_tracker) -> term_info:
    # And the thing we want to call we can now render.
    expr, new_context = render_callable(callable, context, callable.dataframe)

    # In that expr there may be captured variables, or references to things that
    # are not in `value`. If that is the case, that means we need to add a monad to fetch
    # them from earlier in the process.
    root_expr = _find_root_expr(expr, tracker.sequence._ast)
    if root_expr is tracker.sequence._ast:
        # Just continuing on with the sequence already in place.
        assert _is_list(tracker.sequence.result_type)
        # or isinstance(tracker.sequence, statement_unwrap_list)
        # if _is_list(tracker.sequence.result_type):
        #     s, t = _render_expression(
        #         statement_unwrap_list(tracker.sequence._ast, tracker.sequence.result_type),
        #         expr, new_context, tracker)
        # else:
        #     s, t = _render_expression(tracker.sequence, expr, new_context, tracker)
        seq = tracker.sequence.unwrap_if_possible()
        s, t = _render_expression(seq, expr, new_context, tracker)
        assert t.term == 'main_sequence'
        if _is_list(tracker.sequence.result_type):
            s = [smt.wrap() for smt in s]
        if len(s) > 0:
            tracker.statements += s
            tracker.sequence = s[-1]
        return t

    elif root_expr is not None:
        monad_index = tracker.carry_monad_forward(root_expr)
        monad_ref = _monad_manager.new_monad_ref()

        # Create a pointer to the base monad - which is an object
        with tracker.substitute_ast(
                root_expr, _ast_VarRef(term_info(f'{monad_ref}[{monad_index}]', object,
                                       [monad_ref]))):

            # The var we are going to loop over is a pointer to the sequence.
            seq_as_object = tracker.sequence.unwrap_if_possible()
            select_var = new_term(seq_as_object.result_type)
            select_var_rep_ast = _ast_VarRef(select_var)

            with tracker.substitute_ast(tracker.sequence._ast, select_var_rep_ast):
                trm = _resolve_expr_inline(seq_as_object, expr, new_context, tracker)

        result_type = _type_replace(tracker.sequence.result_type, select_var.type, trm.type)
        st = statement_select(a, tracker.sequence.result_type, result_type,
                              select_var, trm)
        if trm.has_monads():
            st.prev_statement_is_monad()
            st.set_monad_ref(monad_ref)

        tracker.statements.append(st)
        tracker.sequence = st
        return term_info('main_sequence', st.result_type)

    else:
        # If root_expr is none, then whatever it is is a constant. So just select it.
        _render_expresion_as_transform(tracker, context, expr)
        return term_info('main_sequence', tracker.sequence.result_type)


class _statement_tracker:
    '''
    Track statements in separate stacks. We have a parent link so we can
    look all the way back in the stack if need be when looking for a replacement.
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

    def substitute_ast(self, source_ast: ast.AST, replace_with: ast.AST) -> ExitStack:
        '''
        Add to the replacement stack. And do it in order.
        '''
        stack = ExitStack()
        a = source_ast
        while a is not None:
            stack.enter_context(replace_an_ast(self, a, replace_with))
            if isinstance(a, ast_Filter):
                a = a.expr
            else:
                a = None
        return stack

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
    Translate a statement that is a series of map or map-like constructs into
    LINQ-style SQL code. Support select, transform, and filter.
    '''
    def __init__(self, base_sequence: statement_base, context: render_context,
                 p_tracker: _statement_tracker):
        '''
        Create a mapper

        Arguments:
            base_sequence           This is the sequence information that we start wit.
        '''
        ast.NodeVisitor.__init__(self)
        _statement_tracker.__init__(self, base_sequence, p_tracker)
        self.context = context

    def visit(self, a: ast.AST):
        '''
        Check to see if the ast we are about to visit is the one that represents the current
        stream. If that is the case, then there is no need to generate any further code here.

        Also check for a simple replacement.
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

        seq_type = self.sequence.result_type
        var_name = new_term(_unwrap_list(seq_type) if _is_list(seq_type) else seq_type)
        with self.substitute_ast(self.sequence._ast,
                                 _ast_VarRef(var_name)):
            # We will put this inside a Where statement, so instead of a sequence,
            # we will be dealing with an unwrapped sequence.
            term = _resolve_expr_inline(self.sequence, a.filter, self.context, self)
            st = statement_where(a, self.sequence.result_type, var_name, term)
            # This is a bit of a kludge
            if len(self.statements) > 0:
                if self.statements[-1].has_monads():
                    st.prev_statement_is_monad()
            self.statements.append(st)
            self.sequence = st

    def visit_Attribute(self, a: ast.Attribute):
        # Get the stream up to the base expression.
        # if a.value is not self.sequence._ast:
        _render_expresion_as_transform(self, self.context, a.value)

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
            return result.AsAwkwardArray(['col1']).value()[default_col_name]
        else:
            return result

    def visit_call_histogram(self, value: ast.AST, args: List[ast.AST],
                             keywords: List[ast.keyword], a: ast.Call):
        '''
        Generate a histogram. We can't do this in the abstract, unfortunately,
        as we have to get the data and apply the numpy.histogram at this upper level.
        '''
        # TODO: This cant be done async as it is written here, so clearly this is broken.
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

        # Render the sequence that gets us to what we want to map over.
        self.visit(value)

        # Now do the call
        _render_callable(a, callable, self.context, self)

    def visit_Call(self, a: ast.Call):
        # Math function calls are treated like expressions
        if isinstance(a.func, ast.Attribute):
            if hasattr(self, f'visit_call_{a.func.attr}'):
                m = getattr(self, f'visit_call_{a.func.attr}')
                m(a.func.value, a.args, a.keywords, a)
            else:
                _render_expresion_as_transform(self, self.context, a.func.value)

                resolved_args = [_resolve_expr_inline(self.sequence.unwrap_if_possible(),
                                                      arg, self.context, self)
                                 for arg in a.args]
                name = a.func.attr
                self.append_call(a, name, [r.term for r in resolved_args])

        elif isinstance(a.func, ast_FunctionPlaceholder):
            # We will embed this in a select statement. And the sequence items will
            # need to be explicitly referenced in the arguments.

            def do_resolve(arg):
                return _resolve_expr_inline(self.sequence, arg, self.context, self)

            func = cast(ast_FunctionPlaceholder, a.func)
            name = func.callable.__name__
            func_return_type = inspect.signature(func.callable).return_annotation
            if func_return_type is inspect.Signature.empty:
                raise Exception(f"User Error: Function {name} needs return type python hints.")

            var_name = new_term(_unwrap_if_possible(self.sequence.result_type))
            return_seq_type = _type_replace(self.sequence.result_type, var_name.type,
                                            func_return_type)

            with self.substitute_ast(self.sequence._ast,
                                     _ast_VarRef(var_name)):
                resolved_args = [do_resolve(arg) for arg in a.args]

            for t in resolved_args:
                # We can't deal with arrays as arguments yet.
                assert not _is_list(t.type), \
                    f'Functions with array arguments are not supported ({name}) [{t.term}]'
            args = ', '.join(t.term for t in resolved_args)
            monad_refs = set([item for sublist in resolved_args for item in sublist.monad_refs])

            st = statement_select(a, self.sequence.result_type, return_seq_type, var_name,
                                  term_info(f'{name}({args})', func_return_type, list(monad_refs)))
            self.statements.append(st)
            self.sequence = st
        else:
            assert False, 'Function calls can only be method calls or place holders'

    def append_call(self, a: ast.AST, name_of_method: str, args: Optional[List[str]]) -> None:
        'Append a call onto the call chain that will look at this method'
        # Figure out the type information of this function call.

        f_input_type, f_result_type = _type_system(name_of_method)
        result_type = _type_replace(self.sequence.result_type, f_input_type, f_result_type)
        if result_type is None:
            raise RenderException(f'The method "{name_of_method}" requires as input '
                                  f'{str(f_input_type)} but is given {self.sequence.result_type}')

        # Build the function call
        arg_text = "" if args is None else ", ".join([str(ag) for ag in args])
        function_call = f'{name_of_method}({arg_text})'
        iterator_name = new_term(f_input_type)
        expr = f'{iterator_name.term}.{function_call}'

        # Finally, build the call statement as a select, and then update the current sequence.
        select = statement_select(a, self.sequence.result_type, result_type,
                                  iterator_name, term_info(expr, f_result_type))
        self.statements.append(select)
        self.sequence = select


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
                     p_tracker: Optional[_statement_tracker]):
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

            assert (len(s_right) == 0 or not _is_list(s_right[-1].result_type)) \
                or (len(s_left) == 0 or not _is_list(s_left[-1].result_type)), \
                f'double indexes: {ast.dump(a_left)} and {ast.dump(a_right)} ' \
                f'for {ast.dump(operator)}.'

            def do_statements(s: List[statement_base], t: term_info) \
                    -> Tuple[term_info, term_info]:
                if t.term != 'main_sequence':
                    return t, t
                if len(s) == 0:
                    r = new_term(self.sequence.result_type)
                    return r, r
                if not _is_list(s[-1].result_type):
                    stem = new_term(self.sequence.result_type)
                    r = stem
                    for single in s:
                        stem = single.apply_as_function(stem)
                    return r, stem
                else:
                    # TODO: if we remove this if block, then this look like _resolve_inline.
                    # This tests fail if we do that - understand if we really need this.
                    self.statements += s
                    self.sequence = s[-1]
                    r = new_term(_unwrap_list(self.sequence.result_type))
                    return r, r

            l_l_iter, l_l = do_statements(s_left, left)
            l_r_iter, l_r = do_statements(s_right, right)

            # Is the compare between two "terms" we know, or a sequence?
            op, op_type = _known_operators[type(operator)]
            if left.term != 'main_sequence' and right.term != 'main_sequence':
                self.term_stack.append(term_info(f'({l_l.term} {op} {l_r.term})', op_type))
            else:
                expr = f'({l_l.term} {op} {l_r.term})'
                in_type = self.sequence.result_type
                out_type = _type_replace(in_type, l_l.type, op_type)
                out_type = op_type if out_type is None else out_type
                monads = l_l.monad_refs + l_r.monad_refs
                self.statements.append(statement_select(a, in_type, out_type,
                                                        l_l_iter,
                                                        term_info(expr, op_type, list(monads))))
                self.sequence = self.statements[-1]
                for m in monads:
                    self.sequence.set_monad_ref(m)
                    self.sequence.prev_statement_is_monad()
                self.term_stack.append(term_info('main_sequence', out_type))

        def visit_Compare(self, a: ast.Compare):
            # Need better protection here - if visit doesn't render something
            # even if we are deep in an expression... This will catch internal errors
            # in a production system when someone uses a "new" feature of python we
            # don't have yet. As it stands, it will still cause an assertion failure, it will
            # just potentially be far away from the place the problem actually occurred.
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
            assert len(a.values) == 2, 'Cannot do bool operations more than two operands'
            var_name = new_term(self.sequence.result_type)
            with self.substitute_ast(self.sequence._ast,
                                     _ast_VarRef(var_name)):
                left = _resolve_expr_inline(self.sequence, a.values[0], self.context, self)
                right = _resolve_expr_inline(self.sequence, a.values[1], self.context, self)

            op, op_type = _known_operators[type(a.op)]
            expr = f'({left.term}) {op} ({right.term})'
            monads = list(set(left.monad_refs + right.monad_refs))
            in_type = self.sequence.result_type
            out_type = _type_replace(in_type, _unwrap_list(in_type), bool)
            itr_unwrapped = term_info(var_name.term, _unwrap_list(in_type))
            st = statement_select(a, in_type, out_type, itr_unwrapped,
                                  term_info(expr, bool, monads))
            self.statements.append(st)
            self.sequence = st
            self.term_stack.append(term_info('main_sequence', List[bool]))

        def visit_UnaryOp(self, a: ast.UnaryOp):
            'invert'
            assert type(a.op) is ast.Invert, f'Unary operator {a.op} is not supported'

            var_name = new_term(self.sequence.result_type)
            with self.substitute_ast(self.sequence._ast,
                                     _ast_VarRef(var_name)):
                operand = _resolve_expr_inline(self.sequence, a.operand, self.context, self)

            expr = f'not ({operand.term})'
            in_type = self.sequence.result_type
            out_type = _type_replace(in_type, _unwrap_list(in_type), bool)
            itr_unwrapped = term_info(var_name.term, _unwrap_list(in_type))
            st = statement_select(a, in_type, out_type, itr_unwrapped, term_info(expr, bool,
                                  operand.monad_refs))
            self.statements.append(st)
            self.sequence = st
            self.term_stack.append(term_info('main_sequence', List[bool]))

        def visit_Num(self, a: ast.Num):
            'A number term should be pushed into the stack'
            self.term_stack.append(term_info(str(a.n), float))

        def visit_Str(self, a: ast.Str):
            'A string should be pushed onto the stack'
            self.term_stack.append(term_info(f'"{a.s}"', str))

        def visit__ast_VarRef(self, a: _ast_VarRef):
            self.term_stack.append(a.term)

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
            self.term_stack.append(term_info("main_sequence", self.sequence.result_type))

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
            assert isinstance(a.func, (ast.Attribute, ast_FunctionPlaceholder, ast_Callable))

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
                                      v.type, v.monad_refs))
                    else:
                        var_name = new_term(float)
                        expr = f'({_known_simple_math_functions[a.func.attr]}({var_name.term}))'
                        # An exception will be thrown here if the input sequence isn't float.
                        self.statements.append(
                            statement_select(a, self.sequence.result_type,
                                             self.sequence.result_type,
                                             var_name, term_info(expr, float)))
                        self.term_stack.append(term_info('main_sequence',
                                                         self.sequence.result_type))
            elif isinstance(a.func, ast_Callable):
                assert len(a.args) == 1, "internal error - expect one arg for top level lambda"

                # Render the sequence that gets us to what we want to map over.
                self.visit(a.args[0])
                self.term_stack.pop()

                # And the thing we want to call we can now render.
                func = cast(ast_Callable, a.func)
                t = _render_callable(a, func, self.context, self)
                self.term_stack.append(t)
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
        r.term_stack[0] if len(r.term_stack) != 0 \
        else term_info('main_sequence',
                       r.statements[-1].result_type if len(r.statements) > 0
                       else current_sequence.result_type)


# Known operators and their "text" rep, and the default return type
# of their calculation.
_known_operators: Dict[Type, Tuple[str, Type]] = {
    ast.Div: ('/', float),
    ast.Sub: ('-', float),
    ast.Mult: ('*', float),
    ast.Add: ('+', float),
    ast.Gt: ('>', bool),
    ast.GtE: ('>=', bool),
    ast.Lt: ('<', bool),
    ast.LtE: ('<=', bool),
    ast.Eq: ('==', bool),
    ast.NotEq: ('!=', bool),
    ast.And: ('and', bool),
    ast.Or: ('or', bool),
}


def _resolve_expr_inline(curret_sequence: statement_base, expr: ast.AST, context: render_context,
                         p_tracker: _statement_tracker) \
        -> term_info:
    '''
    Resolve an expression in-line.

    Arguments:
        current_sequence        The current sequence is what - we can grab this from outside
        expr                    Expression representing the argument
        context                 The render context to pass through in case rendering is needed.
        p_tracker               Parent tracker to keep the chain for lookups going
    '''
    # We need to drive the select/where inline generation.
    # 1. object (implied sequence, one item per event):
    #       d.Where(lambda e: e > 10)
    # 2. List[object] (explicit sequence, one list per event):
    #       d.Select(lambda e: e.Where(lambda ep: ep > 10))
    # if _is_list(curret_sequence.result_type):
    #     # Since this guy is a sequence, we have to turn it into not-a sequence for processing.
    #     filter_sequence, trm = _render_expression(
    #         statement_unwrap_list(curret_sequence._ast, curret_sequence.result_type), expr,
    #         context, p_tracker)
    #     # act_on_sequence = True
    # else:
    filter_sequence, trm = _render_expression(curret_sequence, expr, context, p_tracker)

    assert (trm.term != 'main_sequence' and len(filter_sequence) == 0) \
        or (trm.term == 'main_sequence' and len(filter_sequence) > 0)

    if len(filter_sequence) > 0:
        # If we have to create a new variable here, then probably something has gone wrong.
        a_resolved = p_tracker.lookup_ast(curret_sequence._ast)
        assert (a_resolved is not None) \
            or (filter_sequence[0]
                .apply_as_function(term_info('bogus',
                                   filter_sequence[0]._input_sequence_type))
                .term.find('bogus') < 0)
        assert a_resolved is None or isinstance(a_resolved, _ast_VarRef)

        base_type = a_resolved.term.type if a_resolved is not None else object
        base_name = a_resolved.term.term if a_resolved is not None else 'bogus'
        monads = a_resolved.term.monad_refs if a_resolved is not None else []
        stem = term_info(base_name, base_type, monads)
        if _is_list(curret_sequence.result_type) \
                and all(_is_list(s.result_type) for s in filter_sequence):
            prep_statement = [sw.unwrap_if_possible() for sw in filter_sequence]
        else:
            prep_statement = filter_sequence
        for s in prep_statement:
            # TODO: KLUDGE!!!!
            s1 = cast(_monad_manager, s)
            s1._previous_statement_monad = False
            s1._monads = []
            stem = s.apply_as_function(stem)
        return stem
    else:
        return trm


_known_types = {
    'Jets': (object, List[object]),
    'Electrons': (object, List[object]),
    'TruthParticles': (object, List[object]),
    'Count': (List[object], int),
    'First': (List[object], object),
    'tracks': (object, List[object]),
    'jets': (object, List[object]),
    'mcs': (object, List[object]),
    'hasProdVtx': (object, bool),
    'hasDecayVtx': (object, bool),
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
    return _known_types[n] if n in _known_types else (object, float)


# List of math functions we translate into something similar in the LINQ code.
_known_simple_math_functions = {
    'abs': 'abs',
    'sqrt': 'sqrt',
    # numpy functions
    'absolute': 'abs',
}
