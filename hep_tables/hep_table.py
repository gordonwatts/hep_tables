# Top level file for hep_table
from dataframe_expressions import DataFrame
from func_adl import EventDataset


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

        # Check the arguments
        if len(events) == 0:
            raise Exception('xaod_table must be created with an EventDataset derived data source')

        for s in events:
            if not isinstance(s, EventDataset):
                raise Exception(f'xaod_table can only work with EventDataset derived data sources: {s}')

        self.event_source = events

    def __deepcopy__(self, memo):
        '''
        Specialize the deep copy, as the event source represents
        (or may) a resource.
        '''
        return xaod_table(*self.event_source)
