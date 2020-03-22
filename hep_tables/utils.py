import ast
import logging
from typing import Any, List, Type, Optional, Dict

from dataframe_expressions import DataFrame, ast_DataFrame, ast_Filter, render
from func_adl_xAOD import use_exe_servicex
from func_adl import ObjectStream

from .hep_table import xaod_table

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


def _is_sequence(n: str):
    'Determine if the call on n is a collection or a terminal'
    return (n == 'jets') or (n == 'Jets') or (n == 'Electrons')


def _resolve_arg(a: ast.AST) -> str:
    if isinstance(a, ast.Str):
        return f'"{a.s}"'
    if isinstance(a, ast.Num):
        return str(a.n)
    raise Exception("Can only deal with strings and numbers as terminals")


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


def _render_filter_expression(a: ast.AST) -> str:
    '''
    Given an expression that will need no filling in from other places,
    render it to text (so that it can be parsed back again... eye roll).
    '''
    class render_expression(ast.NodeVisitor):
        def __init__(self):
            ast.NodeVisitor.__init__(self)
            self.term_stack = []

        def visit_Compare(self, a: ast.Compare):
            assert len(a.comparators) == 1
            assert len(a.ops) == 1
            # Need better protection here - if visit doesn't render something
            # even if we are deep in an expression... This will catch internal errors
            # in a production system when someone ueses a "new" feature of python we
            # don't have yet. As it stands, it will still cause an assertion failure, it will
            # just potentially be far away from the place the problem actually occured.
            self.visit(a.left)
            left = self.term_stack.pop()
            self.visit(a.comparators[0])
            right = self.term_stack.pop()
            if type(a.ops[0]) not in _known_operators:
                raise Exception(f'Unknown operator {str(a.ops[0])} - cannot translate.')
            op = _known_operators[type(a.ops[0])]
            self.term_stack.append(f'({left} {op} {right})')

        def visit_Name(self, a: ast.Name):
            self.term_stack.append(a.id)

        def visit_Num(self, a: ast.Num):
            self.term_stack.append(str(a.n))

        def visit_Str(self, a: ast.Str):
            self.term_stack.append(f'"{a.s}"')

    r = render_expression()
    r.visit(a)
    assert len(r.term_stack) == 1
    return r.term_stack.pop()


class _map_to_data(ast.NodeVisitor):
    def __init__(self):
        self.dataset: Optional[xaod_table] = None
        self.call_chain: List[seq_info] = []
        self._counter = 1
        self._seen_asts: Dict[int, str] = {}

    def visit(self, a: ast.AST):
        # This is a horrible hack that will have to be smarter when we do object
        # to object comparisons. But we aren't trying to get that working yet.
        # so why spend the effort?
        ast.NodeVisitor.visit(self, a)
        self._seen_asts[hash(str(a))] = 'base_value'

    def new_name(self) -> str:
        n = f'e{self._counter}'
        self._counter += 1
        return n

    def visit_ast_DataFrame(self, a: ast_DataFrame):
        df = a.dataframe
        assert isinstance(df, xaod_table), "Can only use xaod_table dataframes in a query"
        self.dataset = df.event_source
        self.call_chain.append(seq_info(lambda a: self.dataset, xaod_table))

    def visit_ast_Filter(self, a: ast_Filter):
        self.visit(a.expr)
        # Get any references we already know about and replace them.

        class replace_known(ast.NodeTransformer):
            def __init__(self, known: Dict[int, str]):
                ast.NodeTransformer.__init__(self)
                self.known = known
                pass

            def visit(self, a: ast.AST):
                h = hash(str(a))
                if h in self.known:
                    return ast.Name(self.known[h])

                return ast.NodeTransformer.visit(self, a)

        replaced_a = replace_known(self._seen_asts).visit(a.filter)
        filter_expression = _render_filter_expression(replaced_a)
        self.append_filter(filter_expression)

    def append_filter(self, expr: str):
        'Put in a Where statement'
        result_type = self.call_chain[-1].type
        working_on_sequence = result_type is List[object]
        if working_on_sequence:
            v_name = self.new_name()
            s_name = self.new_name()
            expr_to_call = expr.replace('base_value', s_name)
            self.call_chain \
                .append(seq_info(lambda a: a.Select(f"lambda {v_name}: {v_name}"
                                                    f".Where(lambda {s_name}: {expr_to_call})"),
                                 result_type))
        else:
            v_name = self.new_name()
            expr_to_call = expr.replace('base_value', v_name)
            self.call_chain.append(seq_info(lambda a: a.Where(f"lambda {v_name}: {expr_to_call}"),
                                            result_type))

    def append_expression(self, expr: str, result_type: Type) -> None:
        '''
        Append an expression.

        Arguments:
            expr            String expression. Contains the string base_value which will be
                            replaced with whateve the current object reference is.
            result_type     What type of object will this yield.
        '''
        # TODO: Proper way to deal with typeing in python when we use it for introspection.
        # THis is only working now b.c. we are doing "object" as a thing
        working_on_sequence = self.call_chain[-1].type is List[object]
        if working_on_sequence:
            # If this is a sequence, then we will keep it a sequence. Further, if this
            # call is going to produce a sequence, then we need to go down one level.
            result_type = List[result_type]
        if working_on_sequence:
            v_name = self.new_name()
            s_name = self.new_name()
            expr_to_call = expr.replace('base_value', s_name)
            self.call_chain \
                .append(seq_info(lambda a: a.Select(f"lambda {v_name}: {v_name}"
                                                    f".Select(lambda {s_name}: {expr_to_call})"),
                                 result_type))
        else:
            v_name = self.new_name()
            expr_to_call = expr.replace('base_value', v_name)
            self.call_chain.append(seq_info(lambda a: a.Select(f"lambda {v_name}: {expr_to_call}"),
                                            result_type))

    def append_call(self, name_of_method: str, args: Optional[List[str]]) -> None:
        'Append a call onto the call chain that will look at this method'
        arg_text = "" if args is None else ", ".join([str(ag) for ag in args])
        function_call = f'{name_of_method}({arg_text})'
        result_type = List[object] if _is_sequence(name_of_method) else object
        self.append_expression(f'base_value.{function_call}', result_type)

    def visit_Attribute(self, a: ast.Attribute):
        self.generic_visit(a)
        name = a.attr
        self.append_call(name, None)

    def visit_Call(self, a: ast.Call):
        assert isinstance(a.func, ast.Attribute), 'Function calls can only be method calls'
        self.visit(a.func.value)

        resolved_args = [_resolve_arg(arg) for arg in a.args]
        name = a.func.attr
        self.append_call(name, resolved_args)

    def visit_BinOp(self, a: ast.BinOp):
        # TODO: Should support 1.0 / j.pt as well as j.pt / 1.0
        if type(a.op) not in _known_operators:
            raise Exception(f'Operator {str(type(a.op))} not known, so cannot translate.')
        operand2 = _resolve_arg(a.right)
        self.visit(a.left)
        self.append_expression(f'base_value {_known_operators[type(a.op)]} {operand2}', object)


def make_local(df: DataFrame) -> Any:
    '''
    Given a dataframe, take its data and render it locally.
    '''
    # First step, get the expression, filter, etc., from the thing.
    expression = render(df)
    lg = logging.getLogger(__name__)
    lg.info(f'make_local expression: {ast.dump(expression)}')

    # Lets render the code to access the data that has been
    # requested.
    mapper = _map_to_data()
    mapper.visit(expression)

    assert mapper.dataset is not None
    result: Optional[ObjectStream] = None
    for c in mapper.call_chain:
        result = c.functor(result)  # type: Optional[ObjectStream]
    assert result is not None
    result = result.AsAwkwardArray(['col1'])

    return result.value(use_exe_servicex)
