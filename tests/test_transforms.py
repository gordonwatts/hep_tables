import ast
from typing import List, Optional
from func_adl.object_stream import ObjectStream, ObjectStreamException
from hep_tables.transforms import sequence_predicate_base


def test_sequence_predicate_base():
    class mtb(sequence_predicate_base):
        def doit(self):
            pass

        def sequence(self, seq: Optional[ObjectStream]) -> ObjectStream:
            return ObjectStream(ast.Constant(10))

        def args(self) -> List[ast.AST]:
            return []

    mtb()
