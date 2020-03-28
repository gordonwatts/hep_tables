import ast
from typing import Dict, List, Optional, Tuple, Type

from dataframe_expressions import ast_DataFrame, ast_Filter
from func_adl_xAOD import use_exe_servicex

from .statements import (
    statement_base, statement_df, statement_select, statement_unwrap_list,
    statement_where, statement_constant)
from .utils import new_var_name, to_args_from_keywords


class _map_to_data(ast.NodeVisitor):
    '''
    The main driver of the translation. We build up a LINQ query - this means we have a
    "sequence" that we are manipulating as we move through by either transforming (with
    Select) or filtering (with Where). As such, we maintain something that tells us what
    the current sequence is.
    '''
    def __init__(self, base_sequence: statement_base):
        '''
        Create a mapper

        Arguments:
            base_sequence           This is the sequence infomration that we start wit.
        '''
        self.sequence = base_sequence
        self.statements: List[statement_base] = []

    def visit(self, a: ast.AST):
        '''
        Check to see if the ast we are about to visit is the one that represents the current
        stream. If that is the case, then there is no need to generate any further code here.
        '''
        if a is self.sequence._ast:
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
                statement_unwrap_list(self.sequence._ast, self.sequence.rep_type), a.filter)
            act_on_sequence = True
            assert trm == 'main_sequence', 'Unexpected term type for filter expression'
        else:
            filter_sequence, trm = _render_expression(self.sequence, a.filter)
            act_on_sequence = False
            assert trm == 'main_sequence', 'Unexpected term type for filter expression'

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
        # Get the stream up to the base of our expression.
        self.generic_visit(a)

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
        mapper = _map_to_data(statement_df(base_ast_df))
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

    def visit_Call(self, a: ast.Call):
        assert isinstance(a.func, ast.Attribute), 'Function calls can only be method calls'
        # Math function calls are treated like expressions
        if a.func.attr in _known_simple_math_functions:
            r, _ = _render_expression(self.sequence, a)
            self.statements = self.statements + r
        elif hasattr(self, f'visit_call_{a.func.attr}'):
            m = getattr(self, f'visit_call_{a.func.attr}')
            m(a.func.value, a.args, a.keywords, a)
        else:
            # We are now accessing a column or collection off a event or other collection.
            # The LINQ, functional, way of doing this is by going down a level.
            self.visit(a.func.value)

            # TODO: resovle arg should be a call to the expression thing!
            resolved_args = [_resolve_arg(arg) for arg in a.args]
            name = a.func.attr
            self.append_call(a, name, resolved_args)

    def visit_BinOp(self, a: ast.BinOp):
        # TODO: Should support 1.0 / j.pt as well as j.pt / 1.0
        r, _ = _render_expression(self.sequence, a)
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


def _render_expression(current_sequence: statement_base, a: ast.AST) \
        -> Tuple[List[statement_base], str]:
    '''
    Render an expression. If the expression contains linq selections stuff, then we will
    have to go back and render those as well by making a call back into _map.

    Arguments:
        current_sequence        The current active sequence that we are looking at
        a                       The expression ast to be parsed

    Returns:
        statements              A list of statements to be appended or referenced from the current
                                sequence.
        term                    'main_sequence' or a string term representing the expression

    '''
    class render_expression(ast.NodeVisitor):
        def __init__(self, current_sequence: statement_base):
            ast.NodeVisitor.__init__(self)
            self.sequence = current_sequence
            self.statements = []
            self.term_stack = []

        def binary_op_statement(self, operator: ast.AST, a_left: ast.AST, a_right: ast.AST):
            '''
            Create the statements needed for a binary operator
            '''
            s_left, left = _render_expression(self.sequence, a_left)
            self.statements += s_left

            s_right, right = _render_expression(self.sequence, a_right)
            self.statements += s_right

            # Is the compare between two "terms" we know, or a sequence?
            op = _known_operators[type(operator)]
            if left != 'main_sequence':
                self.term_stack.append(f'({left} {op} {right})')
            else:
                var_name = new_var_name()
                expr = f'({var_name} {op} {right})'
                rep_type = s_left[-1].rep_type if len(s_left) > 0 else self.sequence.rep_type
                self.statements.append(statement_select(a, bool, var_name, expr,
                                                        rep_type is List[object]))
                self.term_stack.append('main_sequence')

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
                s_left, left = _render_expression(r.sequence, a)
                # r.statements += s_left
                return s_left, left

            assert len(a.values) == 2
            terms = [process_term(self, a) for a in a.values]
            assert [t[1] for t in terms].count('main_sequence') == len(terms)

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
            self.term_stack.append('main_sequence')

        def visit_Num(self, a: ast.Num):
            'A number term should be pushed into the stack'
            self.term_stack.append(str(a.n))

        def visit_Str(self, a: ast.Str):
            'A string should be pushed onto the stack'
            self.term_stack.append(f'"{a.s}"')

        def process_with_mapper(self, a: ast.AST):
            '''
            Use the main reducer to parse this ast.
            '''
            mapper = _map_to_data(self.sequence)
            mapper.visit(a)
            if len(mapper.statements) > 0:
                self.statements = self.statements + mapper.statements
                self.sequence = self.statements[-1]

            # The stream is now the term we want to use. In order to do this we'll now have
            # to deal with a sequence reference. This is the main sequence, so we want to leave
            # that as the term.
            self.term_stack.append("main_sequence")

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
            assert isinstance(a.func, ast.Attribute)
            if a.func.attr not in _known_simple_math_functions:
                self.process_with_mapper(a)
            else:
                assert len(a.args) == 0

                self.visit(a.func.value)
                v = self.term_stack.pop()

                if v != 'main_sequence':
                    self.term_stack.append(f'{_known_simple_math_functions[a.func.attr]}({v})')
                else:
                    var_name = new_var_name()
                    expr = f'({_known_simple_math_functions[a.func.attr]}({var_name}))'
                    self.statements.append(
                        statement_select(a, object, var_name, expr,
                                         self.sequence.rep_type is List[object]))
                    self.term_stack.append('main_sequence')

    r = render_expression(current_sequence)
    r.visit(a)
    assert len(r.term_stack) == 1
    return r.statements, r.term_stack[0]


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


def _resolve_arg(a: ast.AST) -> str:
    'Hopefully this can be thrown out at some point'
    if isinstance(a, ast.Str):
        return f'"{a.s}"'
    if isinstance(a, ast.Num):
        return str(a.n)
    raise Exception("Can only deal with strings and numbers as terminals")


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
