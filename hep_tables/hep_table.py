# Top level file for hep_table
from hep_tables.exceptions import FuncADLTablesException
from typing import Optional, Type
from dataframe_expressions import DataFrame
from func_adl import EventDataset


class xaod_table (DataFrame):
    '''
    Represents the dataset(s) that will be queried by the array expression.
    '''
    def __init__(self, *events: EventDataset, table_type_info: Optional[Type] = None):
        '''
        A list of `func_adl` data sources that queries can be run against.

        Arguments:

          events        The list of `func_adl` data sources (derived from
                        `func_adl.EventDataSource`). A common example is `ServiceXDatasetSource`.

          table_type_info   The type hint class of the data represented by this table.
        '''
        DataFrame.__init__(self)

        # Check the arguments
        if len(events) == 0:
            raise FuncADLTablesException('xaod_table must be created with an EventDataset derived data source')

        for s in events:
            if not isinstance(s, EventDataset):
                raise FuncADLTablesException(f'xaod_table can only work with EventDataset derived data sources: {s}')

        self.event_source = events
        self._type = table_type_info

    @property
    def table_type(self) -> Type:
        if self._type is None:
            raise FuncADLTablesException('the xaod_table was not supplied with type information!')
        return self._type

    def __deepcopy__(self, memo):
        '''
        Specialize the deep copy, as the event source represents
        (or may) a resource.
        '''
        return self
