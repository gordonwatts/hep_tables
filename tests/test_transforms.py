import ast
from typing import Dict, List, Optional, cast

from func_adl.event_dataset import EventDataset
from func_adl.object_stream import ObjectStream
from func_adl.util_ast import lambda_body

from hep_tables.hep_table import xaod_table
from hep_tables.transforms import (astIteratorPlaceholder, root_sequence_transform, sequence_downlevel,
                                   sequence_predicate_base, sequence_transform, sequence_tuple)

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


def test_tuple_ctor(mocker):
    lst = [
        (ast.Constant(1), mocker.MagicMock(spec=sequence_predicate_base)),
        ([ast.Constant(2), ast.Constant(3)], mocker.MagicMock(spec=sequence_predicate_base)),
    ]
    t = sequence_tuple(lst, 'e101')
    assert len(t.transforms) == 2


def test_tuple_sequence(mocker):
    lst = [
        (cast(ast.AST, ast.Constant(2)), mocker.MagicMock(spec=sequence_predicate_base)),
        (ast.Constant(3), mocker.MagicMock(spec=sequence_predicate_base)),
    ]
    lst[0][1].sequence.return_value = ObjectStream(ast.Name(id='a'))
    lst[1][1].sequence.return_value = ObjectStream(ast.Name(id='b'))

    t = sequence_tuple(lst, 'e1000')
    d: Dict[ast.AST, ast.AST] = {ast.Name(id='hi'): astIteratorPlaceholder()}
    o = ObjectStream(ast.Name('o'))
    assert MatchObjectSequence(o.Select("lambda e1000: (a, b)")) == t.sequence(o, d)

    lst[0][1].sequence.assert_called_with(MatchObjectSequence(ObjectStream(ast.Name(id='e1000'))), d)
    lst[1][1].sequence.assert_called_with(MatchObjectSequence(ObjectStream(ast.Name(id='e1000'))), d)
