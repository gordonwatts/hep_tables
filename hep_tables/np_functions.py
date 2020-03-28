# Numpy functions that we do behind the scenes.
from typing import Optional, Tuple, Union, List
import ast

from dataframe_expressions import DataFrame
from .utils import to_ast


def histogram(df: DataFrame, bins: Union[int, List[float]] = 10,
              range: Optional[Tuple[float, float]] = None,
              density: bool = None) -> DataFrame:
    '''
    Histogram, interface is meant to be the same as regular numpy, as we will use that
    behind the scenes.

    Arguments:
        df              `DataFrame`, single column, that will be used for the data to fill
                        the histogram.
        bins            The number of bins. Passed directly to `numpy.histogram`.
        range           Upper and lower limits on the histogram binning. Passed directly to
                        `numpy.histogram`.
        desnity         Normalize by density or not? Passed directly to `numpy.histogram`.

    Returns:
        DataFrame       The `DataFrame` represents the histogram proxy. If made local, the
                        return is the same as from numpy's hisogram function.
    '''
    keywords = [
        ast.keyword(arg='bins', value=to_ast(bins)),
        ast.keyword(arg='range', value=to_ast(range)),
        ast.keyword(arg="density", value=to_ast(density)),
    ]

    call_node = ast.Call(func=ast.Attribute(value=ast.Name('p'), attr='histogram'),
                         args=[], keywords=keywords)
    return DataFrame(df, call_node)
