

class FuncADLTablesException(Exception):
    '''A generic exception for things that go wrong in hep_tables.
    '''
    def __init__(self, msg):
        super().__init__(self, msg)
