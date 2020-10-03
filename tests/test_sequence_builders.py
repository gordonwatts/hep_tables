import ast
from hep_tables.util_ast import astIteratorPlaceholder
from typing import Callable, Iterable, List, Optional, Type, Union, cast

import pytest
from dataframe_expressions.asts import (ast_Callable, ast_DataFrame,
                                        ast_FunctionPlaceholder)
from dataframe_expressions.data_frame import DataFrame
from dataframe_expressions.render_dataframe import render_context
from igraph import Graph

from hep_tables import xaod_table
from hep_tables.exceptions import FuncADLTablesException
from hep_tables.graph_info import e_info, g_info, get_e_info, get_g_info, get_v_info, v_info
from hep_tables.sequence_builders import ast_to_graph
from hep_tables.transforms import (expression_transform,
                                   root_sequence_transform,
                                   sequence_predicate_base)
from hep_tables.type_info import type_inspector

from .conftest import MatchAST


class Jets:
    def pt(self) -> float:
        ...


class Tracks:
    def pt(self) -> float:
        ...


class TestEvent:
    def ListOfFloats(self) -> List[float]:
        ...

    def AFloat(self) -> float:
        ...

    def Jets(self) -> Iterable[Jets]:
        ...


def test_ast_blank(mocker):
    g = ast_to_graph(ast.Num(n=10))
    assert get_g_info(g) is not None


def test_xaod_table_type(mocker):
    'Add graph node for the root data frame'
    df = mocker.MagicMock(spec=xaod_table)
    df.table_type = TestEvent

    t_mock = mocker.MagicMock(spec=type_inspector)
    g = Graph(directed=True)
    g['info'] = g_info([])

    a = ast_DataFrame(df)
    g = ast_to_graph(a, g, t_mock)

    vertexes = g.vs()
    assert len(vertexes) == 1
    vtx = vertexes[0]
    info = get_v_info(vtx)
    assert info.v_type == Iterable[TestEvent]
    assert info.node is a
    # Make sure we assign a new interator number. g_info resets the counter to zero.
    assert MatchAST('astIteratorPlaceholder(0)') == info.node_as_dict[a]
    assert info.level == 0
    seq = info.sequence
    assert isinstance(seq, root_sequence_transform)
    assert seq.eds is df


def test_xaod_table_not_first(mocker):
    'The xaod_table should be the very first thing added to the graph'
    df = mocker.MagicMock(spec=xaod_table)
    df.table_type = TestEvent

    g = Graph(directed=True)
    a1 = ast_DataFrame(df)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[TestEvent], node={a1: astIteratorPlaceholder(0)}))

    a2 = ast_DataFrame(df)

    t_mock = mocker.MagicMock(spec=type_inspector)
    with pytest.raises(FuncADLTablesException):
        ast_to_graph(a2, g, t_mock)


def test_xaod_table_same_twice(mocker):
    'If we already have a graph entry for the xaod table and we hit the same one again'
    df = mocker.MagicMock(spec=xaod_table)
    df.table_type = TestEvent

    a = ast_DataFrame(df)
    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[TestEvent], node={a: astIteratorPlaceholder(0)}))

    t_mock = mocker.MagicMock(spec=type_inspector)

    g = ast_to_graph(a, g, t_mock)

    vertexes = g.vs()
    assert len(vertexes) == 1


def test_xaod_table_not_xaod_table(mocker):
    'An ast_DataFrame should contain only a xaod_table'
    df = mocker.MagicMock(spec=DataFrame)

    t_mock = mocker.MagicMock(spec=type_inspector)
    g = Graph(directed=True)

    a = ast_DataFrame(df)
    with pytest.raises(FuncADLTablesException):
        ast_to_graph(a, g, t_mock)


def test_attribute_known_list(mocker):
    '''Build an attribute AST for a known concrete type (List[float]), based
    off a top level event iterable.
    '''
    a = ast.Name(id='a')
    pt = ast.Attribute(value=a, attr='AFloat')

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.attribute_type.return_value = Callable[[], float]
    t_mock.callable_type.return_value = ([], float)
    t_mock.iterable_object.return_value = TestEvent

    g = Graph(directed=True)
    a_vtx = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[TestEvent],
                                     {a: astIteratorPlaceholder(2)}))

    ast_to_graph(pt, g, t_mock)

    vertexes = g.vs()
    assert len(vertexes) == 2

    edges = a_vtx.in_edges()
    assert len(edges) == 1

    e1 = edges[0]
    assert get_e_info(e1).main is True
    assert get_e_info(e1).itr_idx == 2
    assert e1.target_vertex == a_vtx

    t_mock.attribute_type.assert_called_once()
    t_mock.attribute_type.assert_called_with(TestEvent, 'AFloat')

    attr_vtx = get_v_info(e1.source_vertex)
    assert attr_vtx.v_type == Iterable[float]
    assert attr_vtx.node is pt
    assert attr_vtx.level == 1
    assert attr_vtx.order == 0
    seq = attr_vtx.sequence
    assert isinstance(seq, expression_transform)

    assert MatchAST("e1000.AFloat()") \
        == seq.render_ast({a: ast.Name(id='e1000')})


def test_attribute_known_list_with_other_parent(mocker):
    '''Build an attribute AST for a known concrete type (List[float]), based
    off a top level event iterable, with a parent. Preventing a regression:
    was going too far back up the chain.
    '''
    a = ast.Name(id='a')
    pt = ast.Attribute(value=a, attr='AFloat')

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.attribute_type.return_value = Callable[[], float]
    t_mock.callable_type.return_value = ([], float)
    t_mock.iterable_object.return_value = TestEvent

    g = Graph(directed=True)
    p_vtx = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[TestEvent],
                                     {ast.Constant(10): astIteratorPlaceholder(10)}))
    a_vtx = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[TestEvent],
                                     {a: astIteratorPlaceholder(8)}))
    g.add_edge(a_vtx, p_vtx, info=e_info(True, 5))

    ast_to_graph(pt, g, t_mock)

    edges = a_vtx.in_edges()
    assert len(edges) == 1

    e1 = edges[0]
    assert get_e_info(e1).main is True
    assert get_e_info(e1).itr_idx == 8


def test_attribute_non_iterable_object(mocker):
    'Try to take the attribute of a non-sequence object, which we cannot handle yet.'
    a = ast.Name(id='a')
    pt = ast.Attribute(value=a, attr='AFloat')

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.attribute_type.return_value = Callable[[], float]
    t_mock.callable_type.return_value = ([], float)
    t_mock.iterable_object.return_value = None

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), TestEvent, {a: astIteratorPlaceholder(1)}))

    with pytest.raises(FuncADLTablesException):
        ast_to_graph(pt, g, t_mock)


def test_attribute_with_args_not_given(mocker):
    'Refer to df.jets.pt_arg, where pt_arg requires an argument'
    a = ast.Name(id='a')
    pt = ast.Attribute(value=a, attr='AFloat')

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.attribute_type.return_value = Iterable[float]
    t_mock.callable_type.return_value = (None, None)
    t_mock.iterable_object.return_value = TestEvent

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[TestEvent], {a: astIteratorPlaceholder(1)}))

    with pytest.raises(FuncADLTablesException):
        ast_to_graph(pt, g, t_mock)


def test_attribute_default(mocker):
    'Request an attribute that does not exist in the current model'
    a = ast.Name(id='a')
    pt = ast.Attribute(value=a, attr='AFloat')

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.attribute_type.return_value = Callable[[float], float]
    t_mock.callable_type.return_value = ([float], float)
    t_mock.iterable_object.return_value = TestEvent

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[TestEvent], {a: astIteratorPlaceholder(1)}))

    with pytest.raises(FuncADLTablesException):
        ast_to_graph(pt, g, t_mock)


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

    g = Graph(directed=True)
    g['info'] = g_info([], iter_index=2)
    a_vtx = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[Jets]], {a: astIteratorPlaceholder(1)}))

    ast_to_graph(pt, g, t_mock)

    vertexes = g.vs()
    assert len(vertexes) == 2

    edges = a_vtx.in_edges()
    assert len(edges) == 1

    e1 = edges[0]
    assert get_e_info(e1).main is True
    assert get_e_info(e1).itr_idx == 1
    assert e1.target_vertex == a_vtx

    attr_vtx = get_v_info(e1.source_vertex)
    assert attr_vtx.v_type == Iterable[Iterable[float]]
    assert attr_vtx.node is pt
    assert cast(astIteratorPlaceholder, attr_vtx.node_as_dict[pt]).iterator_number == 2
    assert attr_vtx.level == 2
    seq = attr_vtx.sequence
    assert isinstance(seq, expression_transform)

    assert MatchAST("e1000.pt()") == seq.render_ast({a: ast.Name(id='e1000')})


def test_attribute_second_to_the_party(mocker):
    'Add a second dependent leaf to a node and make sure it reuses the original iterator'
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

    g = Graph(directed=True)
    a_vtx = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[Jets]], {a: astIteratorPlaceholder(1)}))
    a_leaf = g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[float]], {ast.Num(n=22): astIteratorPlaceholder(3)}))
    g.add_edge(a_leaf, a_vtx, info=e_info(True, 1))

    ast_to_graph(pt, g, t_mock)

    edges = a_vtx.in_edges()
    assert len(edges) == 2

    assert all(get_e_info(e).itr_idx == 1 for e in edges)
    v = list(g.vs())[-1]
    v_info_dict = get_v_info(v).node_as_dict
    assert get_v_info(v).order == 1
    k = list(v_info_dict.keys())[0]
    assert cast(astIteratorPlaceholder, v_info_dict[k]).iterator_number == 3


@pytest.mark.parametrize("operator", [ast.FloorDiv, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.MatMult])
def test_binary_op_unsupported(operator, mocker):
    'Test the binary operators'
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.BinOp(left=a, right=b, op=operator())

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {b: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(op, g, t_mock)

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
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {b: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = (1, (float, float))

    ast_to_graph(op, g, t_mock)

    assert len(g.vs()) == 3
    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[float]
    assert op_v.node is op
    assert op_v.level == 1

    seq = op_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST(f"e1000 {sym} e2000") \
        == seq.render_ast({a: ast.Name(id='e1000'), b: ast.Name(id='e2000')})

    edges = list(g.vs())[-1].out_edges()
    assert all(get_e_info(e).itr_idx == 1 for e in edges)


def test_binary_op_with_parent(mocker):
    'Get iterator index right when we have a deep tree'
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.BinOp(left=a, right=b, op=ast.Add())

    g = Graph(directed=True)
    v_parent = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {ast.Constant(10): astIteratorPlaceholder(1)}))
    v1 = g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[float]], {a: astIteratorPlaceholder(2)}))
    v2 = g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[float]], {b: astIteratorPlaceholder(2)}))
    g.add_edge(v1, v_parent, info=e_info(True, 1))
    g.add_edge(v2, v_parent, info=e_info(True, 1))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = (2, (float, float))

    ast_to_graph(op, g, t_mock)

    edges = list(g.vs())[-1].out_edges()
    assert all(get_e_info(e).itr_idx == 2 for e in edges)


def test_binary_op_cross(mocker):
    'Test the binary operators'
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.BinOp(left=a, right=b, op=ast.Add())

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {b: astIteratorPlaceholder(2)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = (1, (float, float))

    ast_to_graph(op, g, t_mock)

    assert len(g.vs()) == 3
    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[float]
    assert op_v.node is op
    assert op_v.level == 1

    seq = op_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST("e1000 + e2000") \
        == seq.render_ast({a: ast.Name(id='e1000'), b: ast.Name(id='e2000')})

    edges = list(g.vs())[-1].out_edges()
    assert len(edges) == 2
    assert any(get_e_info(e).itr_idx == 1 for e in edges)
    assert any(get_e_info(e).itr_idx == 2 for e in edges)


def test_binary_two_levels_down(mocker):
    'Test binary operator: Iterable[Iterable[float]] + Iterable[Iterable[float]]'
    a = ast.Num('a')
    b = ast.Name('b')
    op = ast.BinOp(left=a, right=b, op=ast.Add())

    # Build the pre-existing graph. If these are from df.jets.pt, then they will already be at level 2, which
    # is where we will need to be operating too - so no change in operator.
    g = Graph(directed=True)
    g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[float]], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[float]], {b: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = (2, (float, float))

    ast_to_graph(op, g, t_mock)

    assert len(g.vs()) == 3
    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[Iterable[float]]
    assert op_v.node is op
    assert op_v.level == 2

    seq = op_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST("e1000 + e2000") \
        == seq.render_ast({a: ast.Name(id='e1000'), b: ast.Name(id='e2000')})
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
    g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[float]], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[Jet]], {b: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = None

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(op, g, t_mock)

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
    r = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {ast.Name('b'): astIteratorPlaceholder(1)}))
    n1 = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[t_left], {a: astIteratorPlaceholder(1)}))
    n2 = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[t_right], {b: astIteratorPlaceholder(1)}))
    g.add_edge(n1, r, info=e_info(True, 1))
    g.add_edge(n2, r, info=e_info(True, 1))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = (1, (t_left, t_right))

    ast_to_graph(op, g, t_mock)

    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[t_result]


def test_function_single_arg(mocker):
    'Call a function with a single argument, which is the main sequence'
    a = ast.Name('a')
    c = ast.Call(func=ast.Name(id='my_func'), args=[a], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[float], float]
    t_mock.callable_type.return_value = ([float], float)
    t_mock.find_broadcast_level_for_args.return_value = (1, (float,))

    ast_to_graph(c, g, t_mock)

    assert len(g.vs()) == 2
    v = list(g.vs())[1]
    call_v = get_v_info(v)

    assert call_v.v_type == Iterable[float]
    assert call_v.node is c
    assert call_v.level == 1

    seq = call_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST("my_func(e1000)") \
        == seq.render_ast({a: ast.Name(id='e1000')})

    t_mock.static_function_type.assert_called_with(g['info'].global_types, "my_func")

    assert len(v.out_edges()) == 1
    assert get_e_info(list(v.out_edges())[0]).itr_idx == 1


def test_function_two_arg(mocker):
    a = ast.Name('a')
    b = ast.Name('b')
    c = ast.Call(func=ast.Name(id='my_func'), args=[a, b], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {b: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[float, float], float]
    t_mock.callable_type.return_value = ([float, float], float)
    t_mock.find_broadcast_level_for_args.return_value = (1, (float, float))

    ast_to_graph(c, g, t_mock)

    assert len(g.vs()) == 3
    call_v = get_v_info(list(g.vs())[-1])

    assert call_v.v_type == Iterable[float]
    assert call_v.node is c
    assert call_v.level == 1

    seq = call_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST("my_func(e1000, e2000)") \
        == seq.render_ast({a: ast.Name(id='e1000'), b: ast.Name(id='e2000')})


def test_function_single_arg_level2(mocker):
    a = ast.Name('a')
    c = ast.Call(func=ast.Name(id='my_func'), args=[a], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    r = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {ast.Name('b'): astIteratorPlaceholder(1)}))
    n1 = g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))
    g.add_edge(n1, r, info=e_info(True, 1))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[float], float]
    t_mock.callable_type.return_value = ([float], float)
    t_mock.find_broadcast_level_for_args.return_value = (2, (float,))

    ast_to_graph(c, g, t_mock)

    assert len(g.vs()) == 3


def test_function_unknown(mocker):
    a = ast.Name('a')
    c = ast.Call(func=ast.Name(id='my_func'), args=[a], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = None

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(c, g, t_mock)

    assert "my_func" in str(e.value)


def test_function_wrong_level(mocker):
    a = ast.Name('a')
    c = ast.Call(func=ast.Name(id='my_func'), args=[a], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[float]], {a: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[float], float]
    t_mock.callable_type.return_value = ([float], float)
    t_mock.find_broadcast_level_for_args.return_value = (2, (float,))

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(c, g, t_mock)

    assert "my_func" in str(e.value)
    assert "dimensions" in str(e.value)


def test_function_number_args(mocker):
    a = ast.Name('a')
    c = ast.Call(func=ast.Name(id='my_func'), args=[a], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[float, float], float]
    t_mock.callable_type.return_value = ([float, float], float)
    t_mock.find_broadcast_level_for_args.return_value = None

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(c, g, t_mock)

    assert "my_func" in str(e.value)


def test_function_placeholder(mocker):
    def func1(a1: float) -> float:
        ...

    a = ast.Name('a')
    c = ast.Call(func=ast_FunctionPlaceholder(func1), args=[a], keywords=[])

    g = Graph(directed=True)
    r = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {ast.Name('b'): astIteratorPlaceholder(1)}))
    n1 = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))
    g.add_edge(n1, r, info=e_info(True, 1))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.callable_signature.return_value = Callable[[float], float]
    t_mock.static_function_type.return_value = Callable[[float], float]
    t_mock.callable_type.return_value = ([float], float)
    t_mock.find_broadcast_level_for_args.return_value = (1, (float,))

    ast_to_graph(c, g, t_mock)

    assert len(g.vs()) == 3
    call_v = get_v_info(list(g.vs())[-1])

    assert call_v.v_type == Iterable[float]
    assert call_v.node is c
    assert call_v.level == 1

    seq = call_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST("func1(e1000)") \
        == seq.render_ast({a: ast.Name(id='e1000')})

    t_mock.callable_signature.assert_called_with(func1, False)


def test_function_placeholder_no_return_type(mocker):
    def func1(a1: float):
        ...

    a = ast.Name('a')
    c = ast.Call(func=ast_FunctionPlaceholder(func1), args=[a], keywords=[])

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.callable_signature.return_value = Callable[[float], None]
    t_mock.callable_type.return_value = ([float], None)
    t_mock.find_broadcast_level_for_args.return_value = (1, (float,))

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(c, g, t_mock)

    assert "return type" in str(e.value)
    assert "func1" in str(e.value)


def test_map(mocker):
    a = ast.Name('a')
    df_a = DataFrame(a)
    map_func = lambda j: j.pt  # noqa
    callable = ast_Callable(map_func, df_a)
    c = ast.Call(func=ast.Attribute(attr='map', value=a), args=[callable])

    g = Graph(directed=True)
    r = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {ast.Name('b'): astIteratorPlaceholder(1)}))
    n1 = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Jets], {a: astIteratorPlaceholder(1)}))
    g.add_edge(n1, r, info=e_info(True, 1))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.attribute_type.return_value = Callable[[], float]
    t_mock.callable_type.return_value = ([], float)
    t_mock.iterable_object.return_value = float

    context = render_context()
    context._lookup_dataframe(df_a)
    context._resolve_ast(a)

    ast_to_graph(c, g, t_mock, context=context)

    assert len(g.vs()) == 3
    call = list(g.vs())[-1]
    call_v = get_v_info(call)

    assert call_v.v_type == Iterable[float]
    # Note - the "node" this refers to is not something we can point to out here.
    # it points to a part of the j.pt in the lambda
    assert call_v.level == 1

    seq = call_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST("e1000.pt()") \
        == seq.render_ast({a: ast.Name(id='e1000')})

    edges = call.out_edges()
    assert len(edges) == 1
    assert get_e_info(edges[0]).itr_idx == 1


def test_double_map(mocker):
    from dataframe_expressions import user_func

    @user_func
    def my_func(a: float, b: float) -> float:
        ...

    a = ast.Name('a')
    b = ast.Name('b')
    df_a = DataFrame(a)
    df_b = DataFrame(b)
    map_func = lambda j: df_b.map(lambda k: j.pt + k.pt)  # noqa
    callable = ast_Callable(map_func, df_a)
    c = ast.Call(func=ast.Attribute(attr='map', value=a), args=[callable])

    g = Graph(directed=True)
    r = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {ast.Name('b'): astIteratorPlaceholder(1)}))
    n1 = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Jets], {a: astIteratorPlaceholder(1)}))
    n2 = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Tracks], {b: astIteratorPlaceholder(2)}))
    g.add_edge(n1, r, info=e_info(True, 1))
    g.add_edge(n2, r, info=e_info(True, 2))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[], float]
    t_mock.callable_type.return_value = ([], float)
    t_mock.iterable_object.return_value = float
    t_mock.find_broadcast_level_for_args.return_value = (1, (float, float))

    context = render_context()
    context._lookup_dataframe(df_a)
    context._lookup_dataframe(df_b)
    context._resolve_ast(a)
    context._resolve_ast(b)

    ast_to_graph(c, g, t_mock, context=context)

    assert len(g.vs()) == 6
    # 1 is the root
    # 2 are n1 and n2
    # 2, added one each to calc the pt
    # 1 added to do the sum

    # The last add should have the two pt ones connected to it.
    add_v = list(g.vs())[-1]
    call_add = get_v_info(add_v)
    assert call_add.v_type == Iterable[float]
    assert len(list(add_v.neighbors(mode='out')))
    left_v, right_v = list(add_v.neighbors(mode='out'))
    assert MatchAST("e1000 + e1001") \
        == call_add.sequence.render_ast({get_v_info(left_v).node: ast.Name(id='e1000'), get_v_info(right_v).node: ast.Name(id='e1001')})

    assert get_v_info(left_v).v_type == Iterable[float]
    assert get_v_info(right_v).v_type == Iterable[float]

    edges = add_v.out_edges()
    assert len(edges) == 2
    assert any(get_e_info(e).itr_idx == 1 for e in edges)
    assert any(get_e_info(e).itr_idx == 2 for e in edges)


# df.jets.map(lambda j: df.tracks.map(lambda t: dr(j.pt, t.pt)))
# df.jets.map(lambda j1: jf.jets.map(lambda j2: dr(j1.pt, j2.pt)))

# TODO: Make sure df.jets.pt() works! and df.jets().pt() too.
