import ast

from dataframe_expressions.render_dataframe import render_context
from hep_tables.graph_info import g_info, get_e_info, get_v_info, v_info

from dataframe_expressions.data_frame import DataFrame
from hep_tables.exceptions import FuncADLTablesException

from func_adl import ObjectStream

from dataframe_expressions.asts import ast_Callable, ast_DataFrame
from hep_tables import xaod_table
from typing import Callable, Iterable, List, Optional, Type, Union

import pytest
from igraph import Graph

from hep_tables.sequence_builders import ast_to_graph
from hep_tables.transforms import astIteratorPlaceholder, root_sequence_transform, sequence_predicate_base, sequence_transform
from hep_tables.type_info import type_inspector
from hep_tables.utils import QueryVarTracker
from .conftest import MatchObjectSequence


class Jets:
    def pt(self) -> float:
        ...


class TestEvent:
    def ListOfFloats(self) -> List[float]:
        ...

    def AFloat(self) -> float:
        ...

    def Jets(self) -> Iterable[Jets]:
        ...


def test_xaod_table_type(mocker):
    'Add graph node for the root data frame'
    df = mocker.MagicMock(spec=xaod_table)
    df.table_type = TestEvent

    t_mock = mocker.MagicMock(spec=type_inspector)
    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    g = Graph(directed=True)

    a = ast_DataFrame(df)
    g = ast_to_graph(a, q_mock, g, t_mock)

    vertexes = g.vs()
    assert len(vertexes) == 1
    vtx = vertexes[0]
    info = get_v_info(vtx)
    assert info.v_type == Iterable[TestEvent]
    assert info.node is a
    assert info.level == 1
    seq = info.sequence
    assert isinstance(seq, root_sequence_transform)
    assert seq.eds is df


def test_xaod_table_not_first(mocker):
    'The xaod_table should be the very first thing added to the graph'
    df = mocker.MagicMock(spec=xaod_table)
    df.table_type = TestEvent

    t_mock = mocker.MagicMock(spec=type_inspector)
    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    g = Graph(directed=True)
    g.add_vertex()

    a = ast_DataFrame(df)
    with pytest.raises(FuncADLTablesException):
        ast_to_graph(a, q_mock, g, t_mock)


def test_xaod_table_not_xaod_table(mocker):
    'An ast_DataFrame should contain only a xaod_table'
    df = mocker.MagicMock(spec=DataFrame)

    t_mock = mocker.MagicMock(spec=type_inspector)
    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    g = Graph(directed=True)

    a = ast_DataFrame(df)
    with pytest.raises(FuncADLTablesException):
        ast_to_graph(a, q_mock, g, t_mock)


def test_attribute_known_list(mocker):
    '''Build an attribute AST for a known concrete type (List[float])
    '''
    a = ast.Name(id='a')
    pt = ast.Attribute(value=a, attr='AFloat')

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.attribute_type.return_value = Callable[[], float]
    t_mock.callable_type.return_value = ([], float)
    t_mock.iterable_object.return_value = TestEvent

    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    q_mock.new_var_name.return_value = 'e1000'

    g = Graph(directed=True)
    a_vtx = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[TestEvent], a))

    ast_to_graph(pt, q_mock, g, t_mock)

    vertexes = g.vs()
    assert len(vertexes) == 2

    edges = a_vtx.in_edges()
    assert len(edges) == 1

    e1 = edges[0]
    assert get_e_info(e1).main is True
    assert e1.target_vertex == a_vtx

    t_mock.attribute_type.assert_called_once()
    t_mock.attribute_type.assert_called_with(TestEvent, 'AFloat')

    attr_vtx = get_v_info(e1.source_vertex)
    assert attr_vtx.v_type == Iterable[float]
    assert attr_vtx.node is pt
    assert attr_vtx.level == 1
    seq = attr_vtx.sequence
    assert isinstance(seq, sequence_transform)

    base = ObjectStream(ast.Name(id='dude'))
    assert MatchObjectSequence(base.Select("lambda e1000: e1000.AFloat()")) == seq.sequence(base, {a: ast.Name(id='e1000')})


def test_attribute_non_iterable_object(mocker):
    'Try to take the attribute of a non-sequence object, which we cannot handle yet.'
    a = ast.Name(id='a')
    pt = ast.Attribute(value=a, attr='AFloat')

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.attribute_type.return_value = Callable[[], float]
    t_mock.callable_type.return_value = ([], float)
    t_mock.iterable_object.return_value = None

    q_mock = mocker.MagicMock(spec=QueryVarTracker)

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), TestEvent, a))

    with pytest.raises(FuncADLTablesException):
        ast_to_graph(pt, q_mock, g, t_mock)


def test_attribute_with_args_not_given(mocker):
    'Refer to df.jets.pt_arg, where pt_arg requires an argument'
    a = ast.Name(id='a')
    pt = ast.Attribute(value=a, attr='AFloat')

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.attribute_type.return_value = Iterable[float]
    t_mock.callable_type.return_value = (None, None)
    t_mock.iterable_object.return_value = TestEvent

    q_mock = mocker.MagicMock(spec=QueryVarTracker)

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[TestEvent], a))

    with pytest.raises(FuncADLTablesException):
        ast_to_graph(pt, q_mock, g, t_mock)


def test_attribute_default(mocker):
    'Request an attribute that does not exist in the current model'
    a = ast.Name(id='a')
    pt = ast.Attribute(value=a, attr='AFloat')

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.attribute_type.return_value = Callable[[float], float]
    t_mock.callable_type.return_value = ([float], float)
    t_mock.iterable_object.return_value = TestEvent

    q_mock = mocker.MagicMock(spec=QueryVarTracker)

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[TestEvent], a))

    with pytest.raises(FuncADLTablesException):
        ast_to_graph(pt, q_mock, g, t_mock)


def test_attribute_implied_loop(mocker):
    'Look at a float number of a list in the list of events'
    a = ast.Name(id='a')
    pt = ast.Attribute(value=a, attr='pt')

    t_mock = mocker.MagicMock(spec=type_inspector)

    def attr_type(attr_type: Type, attr_name: str) -> Optional[Type]:
        if attr_type == Iterable[Jets]:
            return None
        if attr_type == Jets:
            return Callable[[], float]
        assert False, f'called for type {attr_type} - no idea'

    t_mock.attribute_type.side_effect = attr_type
    t_mock.callable_type.return_value = ([], float)

    def iterator_unroll(t: Type) -> Type:
        if t == Iterable[Iterable[Jets]]:
            return Iterable[Jets]
        if t == Iterable[Jets]:
            return Jets
        assert False, "we should not have been called"

    t_mock.iterable_object.side_effect = iterator_unroll

    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    q_mock.new_var_name.return_value = 'e1000'

    g = Graph(directed=True)
    a_vtx = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[Jets]], a))

    ast_to_graph(pt, q_mock, g, t_mock)

    vertexes = g.vs()
    assert len(vertexes) == 2

    edges = a_vtx.in_edges()
    assert len(edges) == 1

    e1 = edges[0]
    assert get_e_info(e1).main is True
    assert e1.target_vertex == a_vtx

    attr_vtx = get_v_info(e1.source_vertex)
    assert attr_vtx.v_type == Iterable[Iterable[float]]
    assert attr_vtx.node is pt
    assert attr_vtx.level == 2
    seq = attr_vtx.sequence
    assert isinstance(seq, sequence_transform)

    base = ObjectStream(ast.Name(id='dude'))
    assert MatchObjectSequence(base.Select("lambda e1000: e1000.pt()")) == seq.sequence(base, {a: ast.Name(id='e1000')})


@pytest.mark.parametrize("operator", [ast.FloorDiv, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.MatMult])
def test_binary_op_unsupported(operator, mocker):
    'Test the binary operators'
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.BinOp(left=a, right=b, op=operator())

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], a))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], b))

    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    t_mock = mocker.MagicMock(spec=type_inspector)

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(op, q_mock, g, t_mock)

    assert "Unsupported" in str(e.value)


@pytest.mark.parametrize("operator, sym", [
                         (ast.Add, '+'),
                         (ast.Sub, '-'),
                         (ast.Mult, '*'),
                         (ast.Div, '/'),
                         (ast.Mod, '%'),
                         (ast.Pow, '**')
                         ])
def test_binary_op(operator, sym, mocker):
    'Test the binary operators'
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.BinOp(left=a, right=b, op=operator())

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], a))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], b))

    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    q_mock.new_var_name.return_value = 'e1000'
    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = (1, (float, float))

    ast_to_graph(op, q_mock, g, t_mock)

    assert len(g.vs()) == 3
    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[float]
    assert op_v.node is op
    assert op_v.level == 1

    seq = op_v.sequence
    assert isinstance(seq, sequence_transform)
    base = ObjectStream(ast.Name(id='dude'))
    assert MatchObjectSequence(base.Select(f"lambda e1000: e1000 {sym} e2000")) \
        == seq.sequence(base, {a: astIteratorPlaceholder(), b: ast.Name(id='e2000')})


def test_binary_two_levels_down(mocker):
    'Test binary operator: Iterable[Iterable[float]] + Iterable[Iterable[float]]'
    a = ast.Num('a')
    b = ast.Name('b')
    op = ast.BinOp(left=a, right=b, op=ast.Add())

    g = Graph(directed=True)
    g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[float]], a))
    g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[float]], b))

    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    q_mock.new_var_name.return_value = 'e1000'
    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = (2, (float, float))

    ast_to_graph(op, q_mock, g, t_mock)

    assert len(g.vs()) == 3
    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[Iterable[float]]
    assert op_v.node is op
    assert op_v.level == 2

    seq = op_v.sequence
    assert isinstance(seq, sequence_transform)
    base = ObjectStream(ast.Name(id='dude'))
    assert MatchObjectSequence(base.Select("lambda e1000: e1000 + e2000")) \
        == seq.sequence(base, {a: astIteratorPlaceholder(), b: ast.Name(id='e2000')})
    t_mock.find_broadcast_level_for_args.assert_called_with((Union[float, int], Union[float, int]),
                                                            (Iterable[Iterable[float]], Iterable[Iterable[float]]))


def test_binary_bad_type(mocker):
    'Test binary operator: Iterable[Iterable[float]] + Iterable[Iterable[Jet]]'
    a = ast.Num('a')
    b = ast.Name('b')
    op = ast.BinOp(left=a, right=b, op=ast.Add())

    class Jet:
        pass

    g = Graph(directed=True)
    g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[float]], a))
    g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[Jet]], b))

    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    q_mock.new_var_name.return_value = 'e1000'
    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = None

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(op, q_mock, g, t_mock)

    assert 'Unable to figure out' in str(e)


@pytest.mark.parametrize("t_left, operator, t_right, t_result", [
                         (float, ast.Add, float, float),
                         (float, ast.Add, int, float),
                         (int, ast.Add, int, int),
                         (float, ast.Sub, float, float),
                         (float, ast.Sub, int, float),
                         (int, ast.Sub, int, int),
                         (float, ast.Mult, float, float),
                         (float, ast.Mult, int, float),
                         (int, ast.Mult, int, int),
                         (float, ast.Div, float, float),
                         (float, ast.Div, int, float),
                         (int, ast.Div, int, float),
                         (float, ast.Mod, float, float),
                         (float, ast.Mod, int, float),
                         (int, ast.Mod, int, int),
                         (float, ast.Pow, float, float),
                         (float, ast.Pow, int, float),
                         (int, ast.Pow, int, int),
                         ])
def test_binary_types(t_left, operator, t_right, t_result, mocker):
    'Test binary operator return types int + float = float, etc.'
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.BinOp(left=a, right=b, op=operator())

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[t_left], a))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[t_right], b))

    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    q_mock.new_var_name.return_value = 'e1000'
    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = (1, (t_left, t_right))

    ast_to_graph(op, q_mock, g, t_mock)

    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[t_result]


def test_function_single_arg(mocker):
    a = ast.Name('a')
    c = ast.Call(func=ast.Name(id='my_func'), args=[a], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], a))

    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    q_mock.new_var_name.return_value = 'e1000'
    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[float], float]
    t_mock.callable_type.return_value = ([float], float)
    t_mock.find_broadcast_level_for_args.return_value = (1, (float,))

    ast_to_graph(c, q_mock, g, t_mock)

    assert len(g.vs()) == 2
    call_v = get_v_info(list(g.vs())[-1])

    assert call_v.v_type == Iterable[float]
    assert call_v.node is c
    assert call_v.level == 1

    seq = call_v.sequence
    assert isinstance(seq, sequence_transform)
    base = ObjectStream(ast.Name(id='dude'))
    assert MatchObjectSequence(base.Select("lambda e1000: my_func(e1000)")) \
        == seq.sequence(base, {a: astIteratorPlaceholder()})

    t_mock.static_function_type.assert_called_with(g['info'].global_types, "my_func")


def test_function_two_arg(mocker):
    a = ast.Name('a')
    b = ast.Name('b')
    c = ast.Call(func=ast.Name(id='my_func'), args=[a, b], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], a))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], b))

    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    q_mock.new_var_name.return_value = 'e1000'
    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[float, float], float]
    t_mock.callable_type.return_value = ([float, float], float)
    t_mock.find_broadcast_level_for_args.return_value = (1, (float, float))

    ast_to_graph(c, q_mock, g, t_mock)

    assert len(g.vs()) == 3
    call_v = get_v_info(list(g.vs())[-1])

    assert call_v.v_type == Iterable[float]
    assert call_v.node is c
    assert call_v.level == 1

    seq = call_v.sequence
    assert isinstance(seq, sequence_transform)
    base = ObjectStream(ast.Name(id='dude'))
    assert MatchObjectSequence(base.Select("lambda e1000: my_func(e1000, e2000)")) \
        == seq.sequence(base, {a: astIteratorPlaceholder(), b: ast.Name(id='e2000')})


def test_function_single_arg_level2(mocker):
    a = ast.Name('a')
    c = ast.Call(func=ast.Name(id='my_func'), args=[a], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], a))

    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    q_mock.new_var_name.return_value = 'e1000'
    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[float], float]
    t_mock.callable_type.return_value = ([float], float)
    t_mock.find_broadcast_level_for_args.return_value = (2, (float,))

    ast_to_graph(c, q_mock, g, t_mock)

    assert len(g.vs()) == 2


def test_function_unknown(mocker):
    a = ast.Name('a')
    c = ast.Call(func=ast.Name(id='my_func'), args=[a], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], a))

    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    q_mock.new_var_name.return_value = 'e1000'
    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = None

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(c, q_mock, g, t_mock)

    assert "my_func" in str(e.value)


def test_function_wrong_level(mocker):
    a = ast.Name('a')
    c = ast.Call(func=ast.Name(id='my_func'), args=[a], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[float]], a))

    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    q_mock.new_var_name.return_value = 'e1000'
    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[float], float]
    t_mock.callable_type.return_value = ([float], float)
    t_mock.find_broadcast_level_for_args.return_value = (2, (float,))

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(c, q_mock, g, t_mock)

    assert "my_func" in str(e.value)
    assert "dimensions" in str(e.value)


def test_function_number_args(mocker):
    a = ast.Name('a')
    c = ast.Call(func=ast.Name(id='my_func'), args=[a], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], a))

    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    q_mock.new_var_name.return_value = 'e1000'
    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[float, float], float]
    t_mock.callable_type.return_value = ([float, float], float)
    t_mock.find_broadcast_level_for_args.return_value = None

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(c, q_mock, g, t_mock)

    assert "my_func" in str(e.value)


def test_map(mocker):
    a = ast.Name('a')
    df_a = DataFrame(a)
    map_func = lambda j: j.pt  # noqa
    callable = ast_Callable(map_func, df_a)
    c = ast.Call(func=ast.Attribute(attr='map', value=a), args=[callable])

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Jets], a))

    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    q_mock.new_var_name.return_value = 'e1000'

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.attribute_type.return_value = Callable[[], float]
    t_mock.callable_type.return_value = ([], float)
    t_mock.iterable_object.return_value = float

    context = render_context()
    context._lookup_dataframe(df_a)
    context._resolve_ast(a)

    ast_to_graph(c, q_mock, g, t_mock, context=context)

    assert len(g.vs()) == 2
    call_v = get_v_info(list(g.vs())[-1])

    assert call_v.v_type == Iterable[float]
    # Note - the "node" this refers to is not something we can point to out here.
    # it points to a part of the j.pt in the lambda
    assert call_v.level == 1

    seq = call_v.sequence
    assert isinstance(seq, sequence_transform)
    base = ObjectStream(ast.Name(id='dude'))
    assert MatchObjectSequence(base.Select("lambda e1000: e1000.pt()")) \
        == seq.sequence(base, {a: astIteratorPlaceholder()})

# df.jets.map(lambda j: df.tracks.map(lambda t: dr(j.pt, t.pt)))
# df.jets.map(lambda j1: jf.jets.map(lambda j2: dr(j1.pt, j2.pt)))

# TODO: Make sure df.jets.pt() works! and df.jets().pt() too.
