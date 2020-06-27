# Top level file for hep_table
from dataframe_expressions import DataFrame


class xaod_table (DataFrame):
    def __init__(self, events):
        # Create the root of the dataframe
        DataFrame.__init__(self)
        self.event_source = events

    def __deepcopy__(self, memo):
        '''
        Specialize the deep copy, as the event source represents
        (or may) a resource.
        '''
        return xaod_table(self.event_source)
