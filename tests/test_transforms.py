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

from .conftest import MatchAST, MatchASTDict, MatchObjectSequence, parse_ast_string


def test_sequence_predicate_base():
    class mtb(sequence_predicate_base):
        def doit(self):
            pass

        def sequence(self, seq: Optional[ObjectStream]) -> ObjectStream:
            return ObjectStream(ast.Constant(10))

        def args(self) -> List[ast.AST]:
            return []


def test_exp_trans_null():
    s = expression_transform(lambda_body(ast.parse("lambda a: 20")))
    assert not s.is_filter


def test_exp_trans_no_args():
    s = expression_transform(lambda_body(ast.parse("lambda a: 20")))
    assert MatchAST("20") == s.render_ast({})


def test_exp_as_filter():
    s = expression_transform(lambda_body(ast.parse("lambda a: a > 20")), is_filter=True)
    assert s.is_filter


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
    'Test a simple constant for the transform - sequence'
    s = mocker.MagicMock(spec=expression_predicate_base)
    s.render_ast.return_value = ast.Num(n=1.0)
    s.is_filter = False
    down = sequence_downlevel(s, "e1000", 1)

    assert down.transform is s

    o = ObjectStream(ast.Name('o'))
    my_dict = {}
    rendered = down.sequence(o, my_dict)

    assert MatchObjectSequence(o.Select("lambda e1000: 1.0")) == rendered
    s.render_ast.assert_called_with(my_dict)


def test_downlevel_one_filter(mocker):
    'Test a simple constant for the transform - sequence - when it is a filter'
    s = mocker.MagicMock(spec=expression_predicate_base)
    s.render_ast.return_value = ast.Num(n=1.0)
    s.is_filter = True
    down = sequence_downlevel(s, "e1000", 1)

    assert down.transform is s

    o = ObjectStream(ast.Name('o'))
    my_dict = {}
    rendered = down.sequence(o, my_dict)

    assert MatchObjectSequence(o.Where("lambda e1000: 1.0")) == rendered
    s.render_ast.assert_called_with(my_dict)


def test_downlevel_one_sequence_list(mocker):
    'Test a simple constant for the transform - sequence'
    s = mocker.MagicMock(spec=expression_predicate_base)
    s.render_ast.return_value = ast.Num(n=1.0)
    s.is_filter = False
    down = sequence_downlevel(s, "e1000", [1])

    assert down.transform is s

    o = ObjectStream(ast.Name('o'))
    my_dict = {}
    rendered = down.sequence(o, my_dict)

    assert MatchObjectSequence(o.Select("lambda e1000: 1.0")) == rendered
    s.render_ast.assert_called_with(my_dict)


def test_downlevel_one_ast(mocker):
    'Test a simple constant for the transform - render_ast'
    s = mocker.MagicMock(spec=expression_predicate_base)
    s.render_ast.return_value = ast.Num(n=1.0)
    s.is_filter = False
    down = sequence_downlevel(s, "e1000", 1, ast.Name('a'))

    my_dict = {}
    rendered = down.render_ast(my_dict)

    assert MatchAST("Select(a, lambda e1000: 1.0)") == rendered
    s.render_ast.assert_called_with(my_dict)


def test_downlevel_with_index_one(mocker):
    'Pass in a single index down'
    s = mocker.MagicMock(spec=expression_predicate_base)
    a_ref = ast.Num(n=1.0)
    s.render_ast.return_value = ast.Name('e1001')
    s.is_filter = False
    down = sequence_downlevel(s, "e1001", 1, a_ref)

    my_dict: Dict[ast.AST, ast.AST] = {a_ref: astIteratorPlaceholder(1, [0])}
    rendered = down.render_ast(my_dict)

    assert MatchAST("Select(astIteratorPlaceholder(1, [0]), lambda e1001: e1001)") == rendered
    s.render_ast.assert_called_with(MatchASTDict({a_ref: "astIteratorPlaceholder(1, [])"}))


def test_downlevel_with_2_index_one(mocker):
    'Pass in a single index down'
    s = mocker.MagicMock(spec=expression_predicate_base)
    a_ref = ast.Num(n=1.0)
    b_ref = ast.Num(n=2.0)
    s.render_ast.return_value = ast.Name('e1001')
    s.is_filter = False
    down = sequence_downlevel(s, "e1001", [1, 2], a_ref)

    my_dict: Dict[ast.AST, ast.AST] = {
        a_ref: astIteratorPlaceholder(1, [0]),
        b_ref: astIteratorPlaceholder(2, [0]),
    }
    rendered = down.render_ast(my_dict)

    assert MatchAST("Select(astIteratorPlaceholder(1, [0]), lambda e1001: e1001)") == rendered
    s.render_ast.assert_called_with(MatchASTDict({
        a_ref: "astIteratorPlaceholder(1, [])",
        b_ref: "astIteratorPlaceholder(2, [])",
    }))


def test_downlevel_with_index_two(mocker):
    'Pass in a single index down'
    s = mocker.MagicMock(spec=expression_predicate_base)
    a_ref = ast.Num(n=1.0)
    s.render_ast.return_value = ast.Subscript(value=ast.Name('e1001'), slice=ast.Index(value=ast.Num(n=0)))
    s.is_filter = False
    down = sequence_downlevel(s, "e1001", 1, a_ref)

    my_dict: Dict[ast.AST, ast.AST] = {a_ref: astIteratorPlaceholder(1, [0, 1])}
    rendered = down.render_ast(my_dict)

    assert MatchAST("Select(astIteratorPlaceholder(1, [0, 1]), lambda e1001: e1001[0])") == rendered
    s.render_ast.assert_called_with(MatchASTDict({a_ref: "astIteratorPlaceholder(1, [0])"}))


def test_downlevel_not_right_itr_index(mocker):
    'Pass in a single index down'
    s = mocker.MagicMock(spec=expression_predicate_base)
    a_ref = ast.Num(n=1.0)
    s.render_ast.return_value = ast.Subscript(value=ast.Name('e1001'), slice=ast.Index(value=ast.Num(n=0)))
    s.is_filter = False
    down = sequence_downlevel(s, "e1001", 2, a_ref)

    my_dict: Dict[ast.AST, ast.AST] = {a_ref: astIteratorPlaceholder(1, [0, 1])}
    rendered = down.render_ast(my_dict)

    assert MatchAST("Select(astIteratorPlaceholder(1, [0, 1]), lambda e1001: e1001[0])") == rendered
    s.render_ast.assert_called_with(MatchASTDict({a_ref: "astIteratorPlaceholder(1, [0,1])"}))


def test_downlevel_two_ast(mocker):
    s = mocker.MagicMock(spec=expression_predicate_base)
    s.render_ast.return_value = parse_ast_string("Select(astIteratorPlaceholder(1,[]), lambda e1000: 1.0)")
    s.is_filter = False
    down = sequence_downlevel(s, "e1001", 1, ast.Name('d'))

    my_dict = {}
    rendered = down.render_ast(my_dict)

    assert MatchAST("Select(d, lambda e1001: Select(e1001, lambda e1000: 1.0))") == rendered
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
