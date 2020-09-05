import ast
from hep_tables.util_ast import astIteratorPlaceholder
from typing import Dict, List, Optional

from func_adl.event_dataset import EventDataset
from func_adl.object_stream import ObjectStream
from func_adl.util_ast import lambda_body

from hep_tables.hep_table import xaod_table
from hep_tables.transforms import (
    expression_predicate_base, expression_transform, expression_tuple,
    root_sequence_transform, sequence_downlevel, sequence_predicate_base)

from .conftest import MatchAST, MatchObjectSequence, parse_ast_string


def test_sequence_predicate_base():
    class mtb(sequence_predicate_base):
        def doit(self):
            pass

        def sequence(self, seq: Optional[ObjectStream]) -> ObjectStream:
            return ObjectStream(ast.Constant(10))

        def args(self) -> List[ast.AST]:
            return []


def test_exp_trans_null():
    expression_transform(lambda_body(ast.parse("lambda a: 20")))


def test_exp_trans_no_args():
    s = expression_transform(lambda_body(ast.parse("lambda a: 20")))
    assert MatchAST("20") == s.render_ast({})


def test_exp_trans_one_args_no_repl():
    a = ast.Num(10)
    s = expression_transform(lambda_body(ast.parse("lambda a: 20")))
    assert MatchAST("20") == s.render_ast({a: ast.Num(30)})


def test_exp_trans_one_args():
    a = ast.Num(10)
    s = expression_transform(a)
    assert MatchAST("30") == s.render_ast({a: ast.Num(30)})


def test_exp_trans_one_args_twice():
    a = ast.Num(10)
    s = expression_transform(a)
    s.render_ast({a: ast.Num(20)})
    assert MatchAST("30") == s.render_ast({a: ast.Num(30)})


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


def test_downlevel_one_sequence(mocker):
    s = mocker.MagicMock(spec=expression_predicate_base)
    s.render_ast.return_value = ast.Num(n=1.0)
    down = sequence_downlevel(s, "e1000")

    assert down.transform is s

    o = ObjectStream(ast.Name('o'))
    my_dict = {}
    rendered = down.sequence(o, my_dict)

    assert MatchObjectSequence(o.Select("lambda e1000: 1.0")) == rendered
    s.render_ast.assert_called_with(my_dict)


def test_downlevel_one_ast(mocker):
    'Downlevel should build a Select statement'
    s = mocker.MagicMock(spec=expression_predicate_base)
    s.render_ast.return_value = ast.Num(n=1.0)
    down = sequence_downlevel(s, "e1000")

    my_dict = {}
    rendered = down.render_ast(my_dict)

    assert MatchAST("Select(astIteratorPlaceholder, lambda e1000: 1.0)") == rendered
    s.render_ast.assert_called_with(my_dict)


def test_downlevel_with_index_Zero(mocker):
    'Downlevel should do an index sub correctly'
    s = mocker.MagicMock(spec=expression_predicate_base)
    a_ref = ast.Num(n=1.0)
    s.render_ast.return_value = ast.Subscript(value=ast.Name('e1001'), slice=ast.Index(value=ast.Num(n=0)))
    down = sequence_downlevel(s, "e1001")

    my_dict: Dict[ast.AST, ast.AST] = {a_ref: astIteratorPlaceholder([0, 1])}
    rendered = down.render_ast(my_dict)

    assert MatchAST("Select(astIteratorPlaceholder, lambda e1001: e1001[0])") == rendered
    c_args = s.render_ast.call_args[0][0]
    assert a_ref in c_args
    v = c_args[a_ref]
    assert isinstance(v, astIteratorPlaceholder)
    assert v.levels == [0]


def test_downlevel_two_ast(mocker):
    s = mocker.MagicMock(spec=expression_predicate_base)
    s.render_ast.return_value = parse_ast_string("Select(astIteratorPlaceholder, lambda e1000: 1.0)")
    down = sequence_downlevel(s, "e1001")

    my_dict = {}
    rendered = down.render_ast(my_dict)

    assert MatchAST("Select(astIteratorPlaceholder, lambda e1001: Select(e1001, lambda e1000: 1.0))") == rendered
    s.render_ast.assert_called_with(my_dict)


def test_tuple_ctor(mocker):
    lst = [
        mocker.MagicMock(spec=expression_predicate_base),
        mocker.MagicMock(spec=expression_predicate_base),
    ]
    t = expression_tuple(lst)
    assert len(t.transforms) == 2


def test_tuple_sequence(mocker):
    lst = [
        mocker.MagicMock(spec=expression_predicate_base),
        mocker.MagicMock(spec=expression_predicate_base),
    ]

    lst[0].render_ast.return_value = ast.Name(id='a')
    lst[1].render_ast.return_value = ast.Name(id='b')

    t = expression_tuple(lst)
    repl_dict: Dict[ast.AST, ast.AST] = {ast.Name(id='a'): ast.Name(id='c')}
    assert MatchAST("(a, b)") == t.render_ast(repl_dict)

    lst[0].render_ast.assert_called_with(repl_dict)
    lst[1].render_ast.assert_called_with(repl_dict)
