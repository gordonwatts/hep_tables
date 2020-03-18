# Top level file for hep_table
from dataframe_expressions import DataFrame


class xaod_table (DataFrame):
    def __init__(self, events):
        # Create the root of the dataframe
        DataFrame.__init__(self)
