from abc import ABC, abstractmethod, abstractproperty
import ast
from typing import List, Optional

from func_adl import EventDataset
from func_adl.object_stream import ObjectStream


class sequence_predicate_base(ABC):
    '''Base class that holds a transform that will convert a stream from one form to another.
    '''
    def __init__(self):
        pass

    @abstractproperty
    def args(self) -> List[ast.AST]:
        raise NotImplementedError()

    @abstractmethod
    def sequence(self, seq: Optional[ObjectStream]) -> ObjectStream:
        raise NotImplementedError()


class sequence_transform(sequence_predicate_base):
    '''Takes a sequence of type L[T1] and translates it to L[T2].
    '''
    def __init__(self):
        pass


class root_sequence_transform(sequence_predicate_base):
    '''Takes the source of all of this'''
    def __init__(self, ds: EventDataset):
        self._eds = ds

    @property
    def eds(self) -> EventDataset:
        '''Return the EventDataSource for this transform

        Returns:
            EventDataset: The root event data source
        '''
        return self._eds

    def args(self) -> List[ast.AST]:
        'There are no arguments for this sequence transform'
        return []

    def sequence(self, seq: Optional[ObjectStream]) -> ObjectStream:
        raise NotImplementedError()
