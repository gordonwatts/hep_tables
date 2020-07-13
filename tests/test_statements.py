import ast
from typing import List

from func_adl import ObjectStream
import pytest

from hep_tables.statements import (
    _monad_manager, statement_base_iterator, statement_select, statement_where, term_info,
    statement_constant)

from hep_tables.utils import _is_of_type, QueryVarTracker


@pytest.fixture
def object_stream(mocker):
    o = mocker.MagicMock(ObjectStream)
    return o


def test_monad_empty():
    m = _monad_manager(QueryVarTracker())
    assert m.render('e1', 'e1.jets()') == 'e1.jets()'


def test_monad_one():
    m = _monad_manager(QueryVarTracker())
    i = m.add_monad('e3', 'e3.eles()')
    assert i == 1
    assert m.render('e1', 'e1.jets()') == '(e1.jets(), e1.eles())'


def test_monad_add_same():
    m = _monad_manager(QueryVarTracker())
    m.add_monad('e3', 'e3.eles()')
    j = m.add_monad('e4', 'e4.eles()')
    assert j == 1
    assert m.render('e1', 'e1.jets()') == '(e1.jets(), e1.eles())'


def test_monad_follow():
    m = _monad_manager(QueryVarTracker())
    index = m.carry_monad_forward(1)
    assert index == 1
    assert m.render('e19', 'e19.jets()') == '(e19[0].jets(), e19[1])'


def test_monad_prev_statement():
    m = _monad_manager(QueryVarTracker())
    m.prev_statement_is_monad()
    assert m.render('e1', 'e1.jets()') == 'e1[0].jets()'


def test_monad_prev_statement_with_monad():
    m = _monad_manager(QueryVarTracker())
    m.add_monad('e3', 'e3[1].eles()')
    m.prev_statement_is_monad()
    assert m.render('e1', 'e1.jets()') == '(e1[0].jets(), e1[1].eles())'


def test_monad_render_with_monad():
    m = _monad_manager(QueryVarTracker())
    m.prev_statement_is_monad()
    assert m.render('(e1, e2)', '(e1, e2).jets()') == 'e1.jets()'


def test_monad_reference_prev():
    m = _monad_manager(QueryVarTracker())
    m.prev_statement_is_monad()
    assert m.render('e1', 'e1.jets(<monad-ref>[1])') == 'e1[0].jets(<monad-ref>[1])'


def test_monad_reference_prev_with_monad():
    m = _monad_manager(QueryVarTracker())
    m.prev_statement_is_monad()
    m.set_monad_ref('<monad-ref>')
    assert m.render('e1', 'e1.jets(<monad-ref>[1])') == 'e1[0].jets(e1[1])'


def test_monad_reference_prev_of_monad():
    m = _monad_manager(QueryVarTracker())
    m.prev_statement_is_monad()
    assert m.render('(e1,e2)', '<monad-ref>[1].jets()') == '<monad-ref>[1].jets()'


def test_monad_reference_prev_of_monad_with_monad():
    m = _monad_manager(QueryVarTracker())
    m.prev_statement_is_monad()
    m.set_monad_ref('<monad-ref>')
    assert m.render('(e1,e2)', '<monad-ref>[1].jets()') == 'e2.jets()'


def test_base_iterator_obj_to_obj():
    a = ast.Num(n=10)
    s = statement_base_iterator(a, object, object,
                                term_info('a', object), term_info('a', object), False,
                                QueryVarTracker())
    r = s._inner_lambda(term_info('b', object), 'Select')
    assert r.term == 'b'
    assert _is_of_type(r.type, object)


def test_base_iterator_int_to_int():
    a = ast.Num(n=10)
    s = statement_base_iterator(a, int, int,
                                term_info('a', int), term_info('a+1', int), False,
                                QueryVarTracker())
    r = s._inner_lambda(term_info('b', int), 'Select')
    assert r.term == 'b+1'
    assert _is_of_type(r.type, int)


def test_base_iterator_List_to_List():
    a = ast.Num(n=10)
    s = statement_base_iterator(a, List[int], List[int],
                                term_info('a', int), term_info('a+1', int), False,
                                QueryVarTracker())
    r = s._inner_lambda(term_info('b', int), 'Select')
    assert r.term == 'b.Select(lambda e0001: e0001+1)'
    assert _is_of_type(r.type, List[int])


def test_base_iterator_List_to_2List():
    a = ast.Num(n=10)
    s = statement_base_iterator(a, List[List[int]], List[List[int]],
                                term_info('a', int), term_info('a+1', int), False,
                                QueryVarTracker())
    r = s._inner_lambda(term_info('b', int), 'Select')
    assert r.term == 'b.Select(lambda e0001: e0001.Select(lambda e0002: e0002+1))'
    assert _is_of_type(r.type, List[List[int]])


def test_base_iterator_Count():
    a = ast.Num(n=10)
    s = statement_base_iterator(a, List[object], int,
                                term_info('a', List[object]), term_info('a.Count()', int), False,
                                QueryVarTracker())
    r = s._inner_lambda(term_info('b', object), 'Select')
    assert r.term == 'b.Count()'
    assert _is_of_type(r.type, int)


def test_base_iterator_Count_Outter_List():
    a = ast.Num(n=10)
    s = statement_base_iterator(a, List[List[object]], int,
                                term_info('a', List[List[object]]), term_info('a.Count()', int),
                                False, QueryVarTracker())
    r = s._inner_lambda(term_info('b', List[object]), 'Select')
    assert r.term == 'b.Count()'
    assert _is_of_type(r.type, int)


def test_base_iterator_Count_Inner_List():
    a = ast.Num(n=10)
    s = statement_base_iterator(a, List[List[object]], List[int],
                                term_info('a', List[object]), term_info('a.Count()', int), False,
                                QueryVarTracker())
    r = s._inner_lambda(term_info('b', List[object]), 'Select')
    assert r.term == 'b.Select(lambda e0001: e0001.Count())'
    assert _is_of_type(r.type, List[int])


def test_select_obj_apply_func_txt_notseq():
    a = ast.Num(n=10)
    rep_type = object
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, float, eb, term_info('eb.pt()', float), QueryVarTracker())

    r = w.apply_as_function(term_info('e5', object))
    assert r.term == 'e5.pt()'
    assert _is_of_type(r.type, float)


def test_select_copies_monads():
    a = ast.Num(n=10)
    rep_type = object
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, float, eb, term_info('eb.pt()', float, ['dude']), QueryVarTracker())
    assert w.has_monad_refs()


def test_select_obj_apply_func_txt_seq():
    a = ast.Num(n=10)
    rep_type = List[object]
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, List[float], eb, term_info('eb.pt()', float), QueryVarTracker())

    r = w.apply_as_function(term_info('e5', List[object]))
    assert r.term == 'e5.Select(lambda e0001: e0001.pt())'
    assert _is_of_type(r.type, List[float])


def test_select_obj_apply_func_unwrap():
    a = ast.Num(n=10)
    rep_type = List[object]
    eb = term_info('eb', object)

    w_base = statement_select(a, rep_type, List[float], eb, term_info('eb.pt()', float), QueryVarTracker())
    w = w_base.unwrap()

    r = w.apply_as_function(term_info('e5', object))
    assert r.term == 'e5.pt()'
    assert _is_of_type(r.type, float)


def test_select_apply_func_unwrap_monad_passthrough():
    a = ast.Num(n=10)
    rep_type = List[object]
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, List[float], eb,
                         term_info('eb.pt()+<monad-ref-1>[0]', float), QueryVarTracker())
    w.prev_statement_is_monad()
    w.set_monad_ref('<monad-ref-1>')

    unwrapped = w.unwrap()

    r = unwrapped.apply_as_function(term_info('e5', object, ['<monad-ref-2>']))

    assert len(r.monad_refs) == 2
    assert '<monad-ref-1>' in r.monad_refs
    assert '<monad-ref-2>' in r.monad_refs


def test_select_apply_func_wrap_monad_passthrough():
    a = ast.Num(n=10)
    rep_type = object
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, float, eb,
                         term_info('eb.pt()+<monad-ref-1>[0]', float), QueryVarTracker())
    w.prev_statement_is_monad()
    w.set_monad_ref('<monad-ref-1>')

    unwrapped = w.wrap()

    r = unwrapped.apply_as_function(term_info('e5', List[object], ['<monad-ref-2>']))

    assert len(r.monad_refs) == 2
    assert '<monad-ref-1>' in r.monad_refs
    assert '<monad-ref-2>' in r.monad_refs


def test_select_obj_apply_func_txt_notseq_prev_monad():
    a = ast.Num(n=10)
    rep_type = object
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, float, eb, term_info('eb.pt()', float), QueryVarTracker())
    w.prev_statement_is_monad()

    r = w.apply_as_function(term_info('e5', object))
    assert r.term == 'e5[0].pt()'
    assert _is_of_type(r.type, float)


def test_select_obj_apply_func_txt_seq_prev_monad():
    a = ast.Num(n=10)
    rep_type = List[object]
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, List[float], eb, term_info('eb.pt()', float), QueryVarTracker())
    w.prev_statement_is_monad()

    r = w.apply_as_function(term_info('e5', List[object]))
    assert r.term == 'e5[0].Select(lambda e0001: e0001.pt())'
    assert _is_of_type(r.type, List[float])


def test_select_obj_apply_func_txt_monad_notseq():
    a = ast.Num(n=10)
    rep_type = object
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, float, eb, term_info('eb.pt()', float), QueryVarTracker())
    w.add_monad('em', 'em.jets()')

    r = w.apply_as_function(term_info('e5', object))
    assert r.term == '(e5.pt(), e5.jets())'
    assert _is_of_type(r.type, float)


def test_select_obj_apply_func_txt_monad_seq():
    a = ast.Num(n=10)
    rep_type = List[object]
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, List[float], eb, term_info('eb.pt()', float), QueryVarTracker())
    w.add_monad('em', 'em.jets()')

    r = w.apply_as_function(term_info('e5', List[object]))
    assert r.term == '(e5.Select(lambda e0001: e0001.pt()), e5.jets())'
    assert _is_of_type(r.type, List[float])


def test_select_apply_func_monad_passed():
    a = ast.Num(n=10)
    rep_type = List[object]
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, List[object], eb,
                         term_info('<monad-ref>[1].pt(eb.index)', object), QueryVarTracker())
    w.prev_statement_is_monad()
    w.set_monad_ref('<monad-ref>')
    r = w.apply_as_function(term_info('e5', List[object]))

    assert r.term == 'e5[0].Select(lambda e0001: <monad-ref>[1].pt(e0001.index))'
    assert _is_of_type(r.type, List[object])
    assert len(r.monad_refs) == 1
    assert r.monad_refs[0] == '<monad-ref>'


def test_select_apply_func_monad_new_sequence():
    a = ast.Num(n=10)
    rep_type = object
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, List[object], eb,
                         term_info('<monad-ref>[1].jets()', List[object]), QueryVarTracker())
    w.prev_statement_is_monad()
    w.set_monad_ref('<monad-ref>')
    r = w.apply_as_function(term_info('e5', object))

    assert r.term == '<monad-ref>[1].jets()'
    assert _is_of_type(r.type, List[object])
    assert len(r.monad_refs) == 1
    assert r.monad_refs[0] == '<monad-ref>'


def test_select_apply_pass_monad_ref_through():
    a = ast.Num(n=10)
    rep_type = List[object]
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, List[float], eb,
                         term_info('eb.pt()', float), QueryVarTracker())
    r = w.apply_as_function(term_info('e5', List[object], ['<monad-ref>']))

    assert r.term == 'e5.Select(lambda e0001: e0001.pt())'
    assert _is_of_type(r.type, List[float])
    assert len(r.monad_refs) == 1
    assert r.monad_refs[0] == '<monad-ref>'


def test_select_apply_gain_monad_ref_through():
    a = ast.Num(n=10)
    rep_type = List[object]
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, List[float], eb,
                         term_info('eb.pt()', float), QueryVarTracker())
    w.prev_statement_is_monad()
    w.set_monad_ref('<monad-ref-1>')
    r = w.apply_as_function(term_info('e5', List[object], ['<monad-ref-2>']))

    assert r.term == 'e5[0].Select(lambda e0001: e0001.pt())'
    assert _is_of_type(r.type, List[float])
    assert '<monad-ref-1>' in r.monad_refs
    assert '<monad-ref-2>' in r.monad_refs
    assert len(r.monad_refs) == 2


def test_select_obj_apply_notseq(object_stream):
    a = ast.Num(n=10)
    rep_type = object
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, float, eb, term_info('eb.pt()', float), QueryVarTracker())

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with('lambda eb: eb.pt()')


def test_select_obj_apply_seq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[object]
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, List[float], eb, term_info('eb.pt()', float), QueryVarTracker())

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with('lambda eb: eb.Select(lambda e0001: e0001.pt())')


def test_select_obj_apply_notseq_prev_monad(object_stream):
    a = ast.Num(n=10)
    rep_type = object
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, float, eb, term_info('eb.pt()', float), QueryVarTracker())
    w.prev_statement_is_monad()

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with('lambda eb: eb[0].pt()')


def test_select_obj_apply_seq_prev_monad(object_stream):
    a = ast.Num(n=10)
    rep_type = List[object]
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, List[float], eb, term_info('eb.pt()', float), QueryVarTracker())
    w.prev_statement_is_monad()

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with('lambda eb: eb[0]'
                                                 '.Select(lambda e0001: e0001.pt())')


def test_select_obj_apply_monad_notseq(object_stream):
    a = ast.Num(n=10)
    rep_type = object
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, float, eb, term_info('eb.pt()', float), QueryVarTracker())
    w.add_monad('em', 'em.jets()')

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with('lambda eb: (eb.pt(), eb.jets())')


def test_select_obj_apply_monad_seq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[object]
    eb = term_info('eb', object)

    w = statement_select(a, rep_type, List[float], eb, term_info('eb.pt()', float), QueryVarTracker())
    w.add_monad('em', 'em.jets()')

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with(
        'lambda eb: (eb.Select(lambda e0001: e0001.pt()), eb.jets())')


# def test_where_apply_func_term_one_level_up():
#     a = ast.Num(n=10)
#     rep_type = List[List[object]]
#     eb = term_info('eb', object)

#     w_base = statement_where(a, rep_type, eb, term_info('eb > 10.0', bool))
#     w = w_base.unwrap()

#     trm = w.apply_as_function(term_info('e10', List[object]))
#     assert trm.term == 'e10 > 10.0'
#     assert _is_of_type(trm.type, List[List[object]])


def test_where_apply_func_seq():
    a = ast.Num(n=10)
    rep_type = List[object]
    eb = term_info('eb', object)

    w = statement_where(a, rep_type, eb, term_info('eb > 10.0', bool), QueryVarTracker())

    trm = w.apply_as_function(term_info('e10', List[object]))
    assert trm.term == 'e10.Where(lambda e0001: e0001 > 10.0)'
    assert _is_of_type(trm.type, List[object])


def test_where_copy_monad_through():
    a = ast.Num(n=10)
    rep_type = List[object]
    eb = term_info('eb', object)

    w = statement_where(a, rep_type, eb, term_info('eb > 10.0', bool, ['dude']), QueryVarTracker())
    assert w.has_monad_refs()


def test_where_apply_func_seq_prev_monad():
    a = ast.Num(n=10)
    rep_type = List[float]
    eb = term_info('eb', float)

    w = statement_where(a, rep_type, eb, term_info('eb > 10.0', bool), QueryVarTracker())
    w.prev_statement_is_monad()

    trm = w.apply_as_function(term_info('e10', List[float]))
    assert trm.term == 'e10[0].Where(lambda e0001: e0001 > 10.0)'
    assert _is_of_type(trm.type, List[float])


def test_where_func_add_monad_seq():
    a = ast.Num(n=10)
    rep_type = List[float]
    eb = term_info('eb', float)

    w = statement_where(a, rep_type, eb, term_info('eb > 10.0', bool), QueryVarTracker())
    index = w.add_monad('em', 'em.jets()')
    assert index == 1

    trm = w.apply_as_function(term_info('e10', List[float]))
    assert trm.term == '(e10.Where(lambda e0001: e0001 > 10.0), e10.jets())'
    assert _is_of_type(trm.type, List[float])


def test_where_pass_func_through_monad():
    a = ast.Num(n=10)
    rep_type = List[int]
    eb = term_info('eb', int)

    f = term_info('eb > <monad-ref>[1]', bool, ['<monad-ref>'])
    w = statement_where(a, rep_type, eb, f, QueryVarTracker())

    trm = w.apply_as_function(term_info('e10', List[int]))
    assert trm.term == 'e10[0].Where(lambda e0001: e0001 > <monad-ref>[1])'
    assert len(trm.monad_refs) == 1


def test_where_obj_apply_notseq(object_stream):
    a = ast.Num(n=10)
    rep_type = object
    eb = term_info('eb', object)

    w = statement_where(a, rep_type, eb, term_info('eb > 10.0', bool), QueryVarTracker())

    w.apply(object_stream)
    object_stream.Where.assert_called_once_with('lambda eb: eb > 10.0')


def test_where_obj_apply_seq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]

    w = statement_where(a, rep_type, term_info('eb', int),
                        term_info('eb > 10.0', bool), QueryVarTracker())

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with('lambda eb: eb.Where(lambda e0001: e0001 > 10.0)')


def test_where_obj_apply_notseq_prev_monad(object_stream):
    a = ast.Num(n=10)
    rep_type = float
    eb = term_info('eb', float)

    w = statement_where(a, rep_type, eb, term_info('eb > 10.0', bool), QueryVarTracker())
    w.prev_statement_is_monad()

    w.apply(object_stream)
    object_stream.Where.assert_called_once_with('lambda eb: eb[0] > 10.0')


def test_where_obj_apply_seq_prev_monad(object_stream):
    a = ast.Num(n=10)
    rep_type = List[int]
    eb = term_info('eb', int)

    w = statement_where(a, rep_type, eb, term_info('eb > 10.0', bool), QueryVarTracker())
    w.prev_statement_is_monad()

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with('lambda eb: eb[0]'
                                                 '.Where(lambda e0001: e0001 > 10.0)')


def test_where_obj_add_monad_noseq(object_stream):
    a = ast.Num(n=10)
    rep_type = float
    eb = term_info('eb', float)

    w = statement_where(a, rep_type, eb, term_info('eb > 10.0', bool), QueryVarTracker())
    index = w.add_monad('em', 'em.jets()')
    assert index == 1

    w.apply(object_stream)
    object_stream.Where.assert_called_once_with('lambda eb: (eb > 10.0, eb.jets())')


def test_where_obj_add_monad_seq(object_stream):
    a = ast.Num(n=10)
    rep_type = List[float]
    eb = term_info('eb', float)

    w = statement_where(a, rep_type, eb, term_info('eb > 10.0', bool), QueryVarTracker())
    index = w.add_monad('em', 'em.jets()')
    assert index == 1

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with(
        'lambda eb: (eb.Where(lambda e0001: e0001 > 10.0), eb.jets())')


def test_where_pass_through_monad(object_stream):
    a = ast.Num(n=10)
    rep_type = List[float]
    eb = term_info('eb', float)

    f = term_info('eb > <monad-ref>[1]', bool, ['<monad-ref>'])
    w = statement_where(a, rep_type, eb, f, QueryVarTracker())

    w.apply(object_stream)
    object_stream.Select.assert_called_once_with(
        'lambda eb: eb[0].Where(lambda e0001: e0001 > eb[1])')


def test_monad_ref_generator():
    m1 = _monad_manager.new_monad_ref()
    assert len(m1) == 8

    m2 = _monad_manager.new_monad_ref()
    assert m1 != m2


def test_constant_obj(object_stream):
    s = statement_constant(ast.Num(n=10), 10, int)
    assert s.apply(object_stream) == 10


def test_constant_func(object_stream):
    s = statement_constant(ast.Num(n=10), 10, int)
    assert s.apply(object_stream) == 10


def test_constant_func_applied(object_stream):
    qvt = QueryVarTracker()
    s = statement_constant(ast.Num(n=10), 10, int)
    t = s.apply_as_function(qvt.new_term(object))
    assert t.term == '10'
    assert t.type is int
