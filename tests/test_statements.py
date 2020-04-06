import ast
from typing import List

from func_adl import ObjectStream
import pytest

from hep_tables.statements import (
    _monad_manager, statement_select, statement_where, term_info)

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


def test_monad_follow():
    m = _monad_manager()
    index = m.carry_monad_forward(1)
    assert index == 1
    assert m.render('e19', 'e19.jets()') == '(e19[0].jets(), e19[1])'


def test_monad_prev_statement():
    m = _monad_manager()
    m.prev_statement_is_monad()
    assert m.render('e1', 'e1.jets()') == 'e1[0].jets()'


def test_monad_prev_statement_with_monad():
    m = _monad_manager()
    m.add_monad('e3', 'e3[1].eles()')
    m.prev_statement_is_monad()
    assert m.render('e1', 'e1.jets()') == '(e1[0].jets(), e1[1].eles())'


def test_monad_render_with_monad():
    m = _monad_manager()
    m.prev_statement_is_monad()
    assert m.render('(e1, e2)', '(e1, e2).jets()') == 'e1.jets()'


def test_monad_reference_prev():
    m = _monad_manager()
    m.prev_statement_is_monad()
    assert m.render('e1', 'e1.jets(<monad-ref>[1])') == 'e1[0].jets(<monad-ref>[1])'


def test_monad_reference_prev_with_monad():
    m = _monad_manager()
    m.prev_statement_is_monad()
    m.set_monad_ref('<monad-ref>')
    assert m.render('e1', 'e1.jets(<monad-ref>[1])') == 'e1[0].jets(e1[1])'


def test_monad_reference_prev_of_monad():
    m = _monad_manager()
    m.prev_statement_is_monad()
    assert m.render('(e1,e2)', '<monad-ref>[1].jets()') == '<monad-ref>[1].jets()'


def test_monad_reference_prev_of_monad_with_monad():
    m = _monad_manager()
    m.prev_statement_is_monad()
    m.set_monad_ref('<monad-ref>')
    assert m.render('(e1,e2)', '<monad-ref>[1].jets()') == 'e2.jets()'


def test_where_obj_apply_notseq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_where(a, rep_type, 'eb', term_info('eb > 10.0', object), False)

    w.apply(object_stream)
    object_stream.Where.assert_called_once_with('lambda eb: eb > 10.0')


def test_where_obj_apply_seq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_where(a, rep_type, 'eb', term_info('eb > 10.0', object), True)

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with('lambda e1: e1.Where(lambda eb: eb > 10.0)')


def test_where_obj_apply_notseq_prev_monad(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_where(a, rep_type, 'eb', term_info('eb > 10.0', object), False)
    w.prev_statement_is_monad()

    w.apply(object_stream)
    object_stream.Where.assert_called_once_with('lambda eb: eb[0] > 10.0')


def test_where_obj_apply_seq_prev_monad(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_where(a, rep_type, 'eb', term_info('eb > 10.0', object), True)
    w.prev_statement_is_monad()

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with('lambda e1: e1[0].Where(lambda eb: eb > 10.0)')


def test_where_obj_add_monad_noseq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_where(a, rep_type, 'eb', term_info('eb > 10.0', object), False)
    index = w.add_monad('em', 'em.jets()')
    assert index == 1

    w.apply(object_stream)
    object_stream.Where.assert_called_once_with('lambda eb: (eb > 10.0, eb.jets())')


def test_where_obj_add_monad_seq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_where(a, rep_type, 'eb', term_info('eb > 10.0', object), True)
    index = w.add_monad('em', 'em.jets()')
    assert index == 1

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with(
        'lambda e1: (e1.Where(lambda eb: eb > 10.0), e1.jets())')


def test_where_pass_through_monad(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    f = term_info('eb > <monad-ref>[1]', object, ['<monad-ref>'])
    w = statement_where(a, rep_type, 'eb', f, True)

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with(
        'lambda e1: e1[0].Where(lambda eb: eb > e1[1])')


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


def test_select_obj_apply_notseq_prev_monad(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', False)
    w.prev_statement_is_monad()

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with('lambda eb: eb[0].pt()')


def test_select_obj_apply_seq_prev_monad(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', True)
    w.prev_statement_is_monad()

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with('lambda e1: e1[0].Select(lambda eb: eb.pt())')


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


def test_select_obj_apply_func_txt_notseq():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', False)

    r = w.apply_as_function(term_info('e5', object))
    assert r.term == 'e5.pt()'


def test_select_obj_apply_func_txt_seq():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', True)

    r = w.apply_as_function(term_info('e5', object))
    assert r.term == 'e5.Select(lambda e1: e1.pt())'


def test_select_obj_apply_func_txt_notseq_prev_monad():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', False)
    w.prev_statement_is_monad()

    r = w.apply_as_function(term_info('e5', object))
    assert r.term == 'e5[0].pt()'


def test_select_obj_apply_func_txt_seq_prev_monad():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', True)
    w.prev_statement_is_monad()

    r = w.apply_as_function(term_info('e5', object))
    assert r.term == 'e5[0].Select(lambda e1: e1.pt())'


def test_select_obj_apply_func_txt_monad_notseq():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', False)
    w.add_monad('em', 'em.jets()')

    r = w.apply_as_function(term_info('e5', object))
    assert r.term == '(e5.pt(), e5.jets())'


def test_select_obj_apply_func_txt_monad_seq():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', True)
    w.add_monad('em', 'em.jets()')

    r = w.apply_as_function(term_info('e5', object))
    assert r.term == '(e5.Select(lambda e1: e1.pt()), e5.jets())'


def test_select_apply_func_manad_passed():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', '<monad-ref>[1].pt(eb.index)', True)
    w.prev_statement_is_monad()
    w.set_monad_ref('<monad-ref>')
    r = w.apply_as_function(term_info('e5', object))

    assert r.term == 'e5[0].Select(lambda e1: <monad-ref>[1].pt(e1.index))'
    assert len(r.monad_refs) == 1
    assert r.monad_refs[0] == '<monad-ref>'


def test_select_apply_pass_monad_ref_through():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', True)
    r = w.apply_as_function(term_info('e5', object, ['<monad-ref>']))

    assert r.term == 'e5.Select(lambda e1: e1.pt())'
    assert len(r.monad_refs) == 1
    assert r.monad_refs[0] == '<monad-ref>'


def test_select_apply_gain_monad_ref_through():
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_select(a, rep_type, 'eb', 'eb.pt()', True)
    w.prev_statement_is_monad()
    w.set_monad_ref('<monad-ref-1>')
    r = w.apply_as_function(term_info('e5', object, ['<monad-ref-2>']))

    assert r.term == 'e5[0].Select(lambda e1: e1.pt())'
    assert '<monad-ref-1>' in r.monad_refs
    assert '<monad-ref-2>' in r.monad_refs
    assert len(r.monad_refs) == 2


def test_monad_ref_generator():
    m1 = _monad_manager.new_monad_ref()
    assert len(m1) == 8

    m2 = _monad_manager.new_monad_ref()
    assert m1 != m2
