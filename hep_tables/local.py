import ast
from hep_tables.hep_table import xaod_table
import logging
from typing import Any

from dataframe_expressions import DataFrame, render
from func_adl_xAOD import use_exe_servicex

from .render import _map_to_data
from .statements import statement_df
from .utils import _find_dataframes


def make_local(df: DataFrame) -> Any:
    '''
    Render a DataFrame's contents locally.

    Arguments:
        df          A DataFrame that is based on an `xaod_table`.

    Returns:
        Values      A Jagged array, or other objects, depending on the query
    '''
    # First step, get the expression, filter, etc., from the thing.
    expression = render(df)
    lg = logging.getLogger(__name__)
    lg.info(f'make_local expression: {ast.dump(expression)}')

    # Find the dataframe on which this is all based.
    base_ast_df = _find_dataframes(expression)
    base_statement = statement_df(base_ast_df)

    # Lets render the code to access the data that has been
    # requested.
    mapper = _map_to_data(base_statement)
    mapper.visit(expression)

    # Render the expression to a LINQ expression.
    # We start with the dataframe.
    # TODO: Dataframe_expressions need dataframe declared as an object stream
    assert isinstance(base_ast_df.dataframe, xaod_table)
    result = base_ast_df.dataframe.event_source
    for seq in mapper.statements:
        result = seq.apply(result)

    return result.AsAwkwardArray(['col1']).value(use_exe_servicex)