import ast
from typing import List, Type, Optional, Dict

from dataframe_expressions import ast_DataFrame, ast_Filter

from .statements import statement_base, statement_select, statement_unwrap_list
from .utils import new_var_name


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
        # self.dataset: Optional[xaod_table] = None
        # self.call_chain: List[seq_info] = []
        # self._counter = 1
        # self._seen_asts: Dict[int, str] = {}

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
        #       d.Select(lambda e: e.Select(labmda ep: ep > 10).Where(lambda f: f))
        if self.sequence.rep_type is List[object]:
            # Since this guy is a sequence, we have to turn it into not-a sequence for processing.
            filter_sequence = _render_expression(statement_unwrap_list(self.sequence._ast,
                                                                       self.sequence.rep_type),
                                                 a.filter)
            # Build the sequence as a series of text strings.
            assert len(filter_sequence) > 0
            var_name = new_var_name()
            stem = var_name
            for s in filter_sequence:
                stem = s.apply_as_text(stem)
            var_name_check = new_var_name()
            expr = f'{stem}.Where(lambda {var_name_check}: {var_name_check})'
            st = statement_select(a, self.sequence.rep_type, var_name, expr, False)
            self.statements.append(st)
            self.sequence = st
        else:
            assert False, "not implemented yet"
        pass

    def visit_Attribute(self, a: ast.Attribute):
        # Get the stream up to the base of our expression.
        self.generic_visit(a)

        # Now we need to "select" a level here. This means doing a call.
        name = a.attr
        self.append_call(a, name, None)

    def visit_Call(self, a: ast.Call):
        assert isinstance(a.func, ast.Attribute), 'Function calls can only be method calls'
        self.visit(a.func.value)

        # TODO: resovle arg should be a call to the expression thing!
        resolved_args = [_resolve_arg(arg) for arg in a.args]
        name = a.func.attr
        self.append_call(a, name, resolved_args)

    def visit_BinOp(self, a: ast.BinOp):
        # TODO: Should support 1.0 / j.pt as well as j.pt / 1.0
        r = _render_expression(self.sequence, a)
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
        result_type = List[object] if _is_sequence(name_of_method) else object
        working_on_sequence = self.sequence.rep_type is List[object]
        if working_on_sequence:
            result_type = List[result_type]

        # Finally, build the map statement, and then update the current sequence.
        select = statement_select(a, result_type, iterator_name, expr, working_on_sequence)
        self.statements.append(select)
        self.sequence = select


def _render_expression(current_sequence: statement_base, a: ast.AST) -> List[statement_base]:
    '''
    Render an expression. If the expression contains linq selections stuff, then we will
    have to go back and render those as well by making a call back into _map.

    Arguments:
        current_sequence        The current active sequence that we are looking at
        a                       The expression ast to be parsed

    Returns:
        statements              A list of statements to be appended or referenced from the current
                                sequence.

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
            self.visit(a_left)
            left = self.term_stack.pop()

            self.visit(a_right)
            right = self.term_stack.pop()

            # Is the compare between two "terms" we know, or a sequence?
            op = _known_operators[type(operator)]
            if left != 'main_sequence':
                self.term_stack.append(f'({left} {op} {right})')
            else:
                var_name = new_var_name()
                expr = f'({var_name} {op} {right})'
                self.statements.append(statement_select(a, bool, var_name, expr,
                                                        self.sequence.rep_type is List[object]))
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

    r = render_expression(current_sequence)
    r.visit(a)
    assert len(r.term_stack) == 1
    assert r.term_stack[0] == 'main_sequence'
    return r.statements


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
    ast.NotEq: '!='
    }


def _resolve_arg(a: ast.AST) -> str:
    'Hopefully this can be thrown out at some point'
    if isinstance(a, ast.Str):
        return f'"{a.s}"'
    if isinstance(a, ast.Num):
        return str(a.n)
    raise Exception("Can only deal with strings and numbers as terminals")


def _is_sequence(n: str):
    'Determine if the call on n is a collection or a terminal'
    return (n == 'jets') or (n == 'Jets') or (n == 'Electrons')
