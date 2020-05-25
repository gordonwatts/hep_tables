import ast
import logging
from typing import Any
from typing import List, Union

from dataframe_expressions import DataFrame, render, Column
from func_adl.ObjectStream import ObjectStream
from func_adl_xAOD import use_exe_servicex
from make_it_sync import make_sync

from hep_tables.hep_table import xaod_table

from .render import _render_expression
from .statements import statement_base, statement_df
from .utils import _find_dataframes

# This is used only when testing is going on.
default_col_name = b'col1'


def _dump_ast(statements: List[statement_base]):
    '''
    Return a string (with new lines) representing what we are about to send
    in.
    '''
    lines = [str(s) for s in statements]
    return '\n'.join(lines)


async def make_local_async(df: Union[DataFrame, Column], force_rerun: bool = False) -> Any:
    '''
    Render a DataFrame's contents locally.

    Arguments:
        df              A DataFrame that is based on an `xaod_table`.
        force_rerun     If true, then no data will be pulled from any cache, and the
                        query will be re-run from scratch.

    Returns:
        Values      A Jagged array, or other objects, depending on the query
    '''
    # First step, get the expression, filter, etc., from the thing.
    expression, context = render(df)
    lg = logging.getLogger(__name__)
    lg.debug(f'make_local expression: {ast.dump(expression)}')

    # Find the dataframe on which this is all based.
    base_ast_df = _find_dataframes(expression)
    base_statement: statement_base = statement_df(base_ast_df)
    assert isinstance(base_ast_df.dataframe, xaod_table)

    # Lets render the code to access the data that has been
    # requested.
    # mapper = _map_to_data(base_statement, context)
    # mapper.visit(expression)

    statements, term = _render_expression(base_statement, expression, context, None)
    assert term.term == 'main_sequence'

    # Render the expression to a LINQ expression.
    # We start with the dataframe.
    # TODO: Dataframe_expressions need dataframe declared as an object stream
    result = base_ast_df.dataframe.event_source
    for seq in statements:
        result = seq.apply(result)

    statement_dump = _dump_ast([base_statement] + statements)
    lg.debug(f'Stem sent to servicex: \n {statement_dump}')

    if isinstance(result, ObjectStream):
        return (await result.AsAwkwardArray(['col1'])
                .value_async(
            lambda a: use_exe_servicex(a, cached_results_OK=not force_rerun)))[default_col_name]
    else:
        return result


make_local = make_sync(make_local_async)
