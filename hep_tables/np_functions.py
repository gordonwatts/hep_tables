# Numpy functions that we do behind the scenes.
from typing import Optional, Tuple

from dataframe_expressions import DataFrame


def histogram(df: DataFrame, bins: int = 10,
              range: Optional[Tuple[float, float]] = None,
              density: bool = None) -> DataFrame:
    '''
    Histogram, interface is meant to be the same as regular numpy, as we will use that
    behind the scenes.
    '''
    return df
    pass
