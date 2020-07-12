import ast
import asyncio
import logging
from typing import Any, List, Union

from dataframe_expressions import Column, DataFrame, render
from dataframe_expressions.render import render_context
from func_adl.ObjectStream import ObjectStream
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


async def _result_from_source_async(s: ObjectStream,
                                    statements: List[statement_base],
                                    base_statement: xaod_table) -> Any:
    'Convert a sequence of statements to its result form for a single source'
    result = s
    for seq in statements:
        result = seq.apply(result)

    statement_dump = _dump_ast([base_statement] + statements)
    lg = logging.getLogger(__name__)
    lg.debug(f'Stem sent to servicex: \n {statement_dump}')

    if isinstance(result, ObjectStream):
        return (await result.AsAwkwardArray(['col1'])
                .value_async())[default_col_name]
    else:
        return result


async def _make_local_from_expression_async(expression: ast.AST,
                                            context: render_context) -> Any:
    # Find the dataframe on which this is all based.
    base_ast_df = _find_dataframes(expression)
    base_statement: statement_base = statement_df(base_ast_df)
    assert isinstance(base_ast_df.dataframe, xaod_table)

    # Next the render
    statements, term = _render_expression(base_statement, expression, context, None)
    assert term.term == 'main_sequence'

    # Render the expressions to a LINQ expression.
    sources = base_ast_df.dataframe.event_source
    results_async = [_result_from_source_async(s, statements, base_statement) for s in sources]

    # Run them all at once.
    results = await asyncio.gather(*results_async)

    if len(results) == 1:
        return results[0]

    import awkward
    return awkward.concatenate(results)


# Unfortunately, this is required to deal with AST NodeVisitors which are not async.
_make_local_from_expression = make_sync(_make_local_from_expression_async)


async def make_local_async(df: Union[DataFrame, Column]) -> Any:  # type: ignore - remove with bug fix in pylance
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

    return await _make_local_from_expression_async(expression, context)


make_local = make_sync(make_local_async)
