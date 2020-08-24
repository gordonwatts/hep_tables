import ast
from typing import List, Optional

from func_adl.event_dataset import EventDataset
from func_adl.object_stream import ObjectStream
from func_adl.util_ast import lambda_body

from hep_tables.hep_table import xaod_table
from hep_tables.transforms import (root_sequence_transform, sequence_downlevel,
                                   sequence_predicate_base, sequence_transform)

from .conftest import MatchObjectSequence


def test_sequence_predicate_base():
    class mtb(sequence_predicate_base):
        def doit(self):
            pass

        def sequence(self, seq: Optional[ObjectStream]) -> ObjectStream:
            return ObjectStream(ast.Constant(10))

        def args(self) -> List[ast.AST]:
            return []

    mtb()


def test_seq_trans_null():
    sequence_transform([ast.Num(20)], "a", lambda_body(ast.parse("lambda a: 20")))


def test_seq_trans_no_args():
    s = sequence_transform([], "a", lambda_body(ast.parse("lambda a: 20")))
    base_seq = ObjectStream(ast.Name(id='dude'))
    new_seq = s.sequence(base_seq, {})
    assert MatchObjectSequence(base_seq.Select("lambda a: 20")) == new_seq


def test_seq_trans_one_args_no_repl():
    a = ast.Num(10)
    s = sequence_transform([a], "a", lambda_body(ast.parse("lambda a: 20")))
    base_seq = ObjectStream(ast.Name(id='dude'))
    new_seq = s.sequence(base_seq, {a: ast.Num(30)})
    assert MatchObjectSequence(base_seq.Select("lambda a: 20")) == new_seq


def test_seq_trans_one_args():
    a = ast.Num(10)
    s = sequence_transform([a], 'b', a)
    base_seq = ObjectStream(ast.Name(id='dude'))
    new_seq = s.sequence(base_seq, {a: ast.Num(30)})
    assert MatchObjectSequence(base_seq.Select("lambda b: 30")) == new_seq


def test_root_sequence_properties(mocker):
    x = mocker.MagicMock(spec=xaod_table)

    t = root_sequence_transform(x)
    assert t.eds is x


def test_root_sequence_apply(mocker):
    x = mocker.MagicMock(spec=xaod_table)
    evt_source = mocker.MagicMock(spec=EventDataset)
    x.event_source = [evt_source]

    t = root_sequence_transform(x)
    r = t.sequence(None, {})
    assert r is evt_source


def test_downlevel_one(mocker):
    s = mocker.MagicMock(spec=sequence_transform)
    s.sequence.side_effect = lambda o, _: o.Select("lambda j: 1.0")
    down = sequence_downlevel(s, "e1000")

    assert down.transform is s
    o = ObjectStream(ast.Name('o'))
    my_dict = {}
    rendered = down.sequence(o, my_dict)

    assert MatchObjectSequence(o.Select("lambda e1000: e1000.Select(lambda j: 1.0)")) == rendered
    s.sequence.assert_called_with(MatchObjectSequence(ObjectStream(ast.Name('e1000'))), my_dict)
