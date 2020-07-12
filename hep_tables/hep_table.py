# Top level file for hep_table
from dataframe_expressions import DataFrame


class xaod_table (DataFrame):
    '''
    Represents the dataset(s) that will be queried by the array expression.
    '''
    def __init__(self, *events):
        '''
        A list of `func_adl` data sources that queries can be run against.

        Arguments:

          events        The list of `func_adl` data sources (derived from
                        `func_adl.EventDataSource`). A common example is `ServiceXDatasetSource`.
        '''
        DataFrame.__init__(self)
        self.event_source = events

    def __deepcopy__(self, memo):
        '''
        Specialize the deep copy, as the event source represents
        (or may) a resource.
        '''
        return xaod_table(self.event_source)
