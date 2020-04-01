import ast
from typing import List

from dataframe_expressions.DataFrame import ast_DataFrame
from func_adl import ObjectStream
import pytest

from hep_tables.hep_table import xaod_table
from hep_tables.statements import (
    _monad_manager, statement_df, statement_select, statement_where)

from .utils_for_testing import f, reset_var_counter  # NOQA


@pytest.fixture
def object_stream(mocker):
    o = mocker.MagicMock(ObjectStream)
    return o


def test_monad_empty():
    m = _monad_manager()
    assert m.render('e1', 'e1.jets()') == 'e1.jets()'


def test_monad_one():
    m = _monad_manager()
    i = m.add_monad('e3', 'e3.eles()')
    assert i == 1
    assert m.render('e1', 'e1.jets()') == '(e1.jets(), e1.eles())'


def test_monad_add_same():
    m = _monad_manager()
    m.add_monad('e3', 'e3.eles()')
    j = m.add_monad('e4', 'e4.eles()')
    assert j == 1
    assert m.render('e1', 'e1.jets()') == '(e1.jets(), e1.eles())'


def test_statement_df_add_monad():
    d = xaod_table(f)
    s = statement_df(ast_DataFrame(d))

    r = s.add_monad('e1', 'e1.jets()')
    assert r is None


def test_where_obj_apply_notseq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_where(a, rep_type, 'eb', 'eb > 10.0', False)

    w.apply(object_stream)
    object_stream.Where.assert_called_once_with('lambda eb: eb > 10.0')


def test_where_obj_apply_seq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_where(a, rep_type, 'eb', 'eb > 10.0', True)

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with('lambda e1: e1.Where(lambda eb: eb > 10.0)')


def test_where_obj_add_monad_noseq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_where(a, rep_type, 'eb', 'eb > 10.0', False)
    index = w.add_monad('em', 'em.jets()')
    assert index == 1

    w.apply(object_stream)
    object_stream.Where.assert_called_once_with('lambda eb: (eb > 10.0, eb.jets())')


def test_where_obj_add_monad_seq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_where(a, rep_type, 'eb', 'eb > 10.0', True)
    index = w.add_monad('em', 'em.jets()')
    assert index == 1

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with(
        'lambda e1: (e1.Where(lambda eb: eb > 10.0), e1.jets())')


def test_select_obj_apply_notseq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', False)

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with('lambda eb: eb.pt()')


def test_select_obj_apply_seq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', True)

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with('lambda e1: e1.Select(lambda eb: eb.pt())')


def test_select_obj_apply_monad_notseq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', False)
    w.add_monad('em', 'em.jets()')

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with('lambda eb: (eb.pt(), eb.jets())')


def test_select_obj_apply_monad_seq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', True)
    w.add_monad('em', 'em.jets()')

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with(
        'lambda e1: (e1.Select(lambda eb: eb.pt()), e1.jets())')


def test_select_obj_apply_txt_notseq():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', False)

    r = w.apply_as_text('e5')
    assert r == 'e5.Select(lambda eb: eb.pt())'


def test_select_obj_apply_txt_seq():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', True)

    r = w.apply_as_text('e5')
    assert r == 'e5.Select(lambda e1: e1.Select(lambda eb: eb.pt()))'


def test_select_obj_apply_txt_monad_notseq():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', False)
    w.add_monad('em', 'em.jets()')

    r = w.apply_as_text('e5')
    assert r == 'e5.Select(lambda eb: (eb.pt(), eb.jets()))'


def test_select_obj_apply_txt_monad_seq():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', True)
    w.add_monad('em', 'em.jets()')

    r = w.apply_as_text('e5')
    assert r == 'e5.Select(lambda e1: (e1.Select(lambda eb: eb.pt()), e1.jets()))'


def test_select_obj_apply_func_txt_notseq():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', False)

    r = w.apply_as_function('e5')
    assert r == 'e5.pt()'


def test_select_obj_apply_func_txt_seq():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', True)

    r = w.apply_as_function('e5')
    assert r == 'e5.Select(lambda e1: e1.pt())'


def test_select_obj_apply_func_txt_monad_notseq():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', False)
    w.add_monad('em', 'em.jets()')

    r = w.apply_as_function('e5')
    assert r == '(e5.pt(), e5.jets())'


def test_select_obj_apply_func_txt_monad_seq():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', True)
    w.add_monad('em', 'em.jets()')

    r = w.apply_as_function('e5')
    assert r == '(e5.Select(lambda e1: e1.pt()), e5.jets())'
