import ast
from typing import Any

from dataframe_expressions import DataFrame, render
import logging


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
    raise Exception("bummer")
