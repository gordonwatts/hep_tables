import ast
import logging
from typing import Any, List

from dataframe_expressions import DataFrame, ast_DataFrame, render
from func_adl_xAOD import use_exe_servicex
from func_adl import ObjectStream

from .hep_table import xaod_table


def _is_sequence(n: str):
    'Determine if the call on n is a collection or a terminal'
    return (n == 'jets') or (n == 'Jets')


def _resolve_arg(a: ast.AST):
    if isinstance(a, ast.Str):
        return f'"{a.s}"'
    if isinstance(a, ast.Num):
        return f'"{a.value}"'
    raise Exception("Can only deal with strings and numbers as arguments to functions")


class _map_to_data(ast.NodeVisitor):
    def __init__(self):
        self.dataset: xaod_table = None
        self.call_chain = []

    def visit_ast_DataFrame(self, a: ast_DataFrame):
        df = a.dataframe
        assert isinstance(df, xaod_table), "Can only use xaod_table dataframes in a query"
        self.dataset = df.event_source
        self.call_chain.append(lambda a: self.dataset)

    def visit_Attribute(self, a: ast.Attribute):
        self.generic_visit(a)
        name = a.attr
        if _is_sequence(name):
            self.call_chain.append(lambda a: a.SelectMany(f"lambda e: e.{name}()"))
        else:
            self.call_chain.append(lambda a: a.Select(f"lambda e: e.{name}()"))

    def visit_Call(self, a: ast.Call):
        assert isinstance(a.func, ast.Attribute), 'Function calls can only be method calls'
        self.visit(a.func.value)

        resolved_args = [_resolve_arg(arg) for arg in a.args]
        name = a.func.attr
        function_call = f'{name}({", ".join([str(ag) for ag in resolved_args])})'
        if _is_sequence(name):
            self.call_chain.append(lambda a: a.SelectMany(f"lambda e: e.{function_call}"))
        else:
            self.call_chain.append(lambda a: a.Select(f"lambda e: e.{function_call}"))


def make_local(df: DataFrame) -> Any:
    '''
    Given a dataframe, take its data and render it locally.
    '''
    # First step, get the expression, filter, etc., from the thing.
    expression, filter = render(df)
    lg = logging.getLogger(__name__)
    lg.info(f'make_local expression: {ast.dump(expression)}')
    lg.info("make_local filter: None" if filter is None
            else f'make_local filter: {ast.dump(filter)}')

    # Lets render the code to access the data that has been
    # requested.
    mapper = _map_to_data()
    mapper.visit(expression)

    assert mapper.dataset is not None
    result: ObjectStream = None
    for c in mapper.call_chain:
        result = c(result)  # type: ObjectStream
    result = result.AsPandasDF(['col1'])

    return result.value(use_exe_servicex)
