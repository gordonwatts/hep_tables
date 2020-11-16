import ast

from hep_tables.constant import Constant
from hep_tables.util_ast import astIteratorPlaceholder
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

import pytest
from dataframe_expressions.asts import (ast_Callable, ast_DataFrame,
                                        ast_FunctionPlaceholder)
from dataframe_expressions.data_frame import DataFrame
from dataframe_expressions.render_dataframe import ast_Filter, render_context
from igraph import Graph

from hep_tables import xaod_table
from hep_tables.exceptions import FuncADLTablesException
from hep_tables.graph_info import e_info, g_info, get_e_info, get_g_info, get_v_info, v_info
from hep_tables.sequence_builders import ast_to_graph
from hep_tables.transforms import (expression_transform,
                                   root_sequence_transform,
                                   sequence_predicate_base)
from hep_tables.type_info import type_inspector

from .conftest import MatchAST, MatchASTDict


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


def test_filter(mocker):
    'Make sure filter works properly for simple case'
    filter = ast.Name('a')
    body = ast.Name('b')
    op = ast_Filter(expr=body, filter=filter)

    g = Graph(directed=True)
    v1 = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[bool], {filter: astIteratorPlaceholder(1)}))
    v2 = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {body: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = ((1, 1), (bool, float))

    ast_to_graph(op, g, t_mock)

    assert len(g.vs()) == 3
    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[float]
    assert op_v.node is op
    assert op_v.level == 1
    assert MatchASTDict({op: astIteratorPlaceholder(1)}) == op_v.node_as_dict

    seq = op_v.sequence
    assert isinstance(seq, expression_transform)
    assert seq.is_filter

    t_mock.find_broadcast_level_for_args.assert_called_with((bool, Any), (Iterable[bool], Iterable[float],))
    assert get_v_info(v2).order != get_v_info(v1).order


def test_filter_bad_test(mocker):
    'Make sure filter works properly for simple case'
    filter = ast.Name('a')
    body = ast.Name('b')
    op = ast_Filter(expr=body, filter=filter)

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {filter: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {body: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = None

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(op, g, t_mock)

    assert "bool" in str(e.value).lower()


@pytest.mark.parametrize("a, c_type", [
                         (ast.Num(n=1), int),
                         (ast.Num(n=1.2), float),
                         (ast.Str(s="hi there"), str),
                         ])
def test_constant(mocker, a, c_type):
    'Make sure we make the proper type for each'
    g = Graph(directed=True)
    t_mock = mocker.MagicMock(spec=type_inspector)

    g = ast_to_graph(a, g, t_mock)

    assert len(g.vs()) == 1
    v = list(g.vs())[0]
    v_info = get_v_info(v)

    assert v_info.v_type == Constant[c_type]  # type: ignore
    assert v_info.node is a


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
    assert e1.target_vertex == a_vtx

    t_mock.attribute_type.assert_called_once()
    t_mock.attribute_type.assert_called_with(TestEvent, 'AFloat')

    attr_vtx = get_v_info(e1.source_vertex)
    assert attr_vtx.v_type == Iterable[float]
    assert attr_vtx.node is pt
    assert attr_vtx.level == 1
    assert attr_vtx.order == 0
    seq = attr_vtx.sequence
    assert MatchASTDict({pt: astIteratorPlaceholder(2)}) == attr_vtx.node_as_dict
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
    g.add_edge(a_vtx, p_vtx, info=e_info(True))

    ast_to_graph(pt, g, t_mock)

    edges = a_vtx.in_edges()
    assert len(edges) == 1

    e1 = edges[0]
    assert get_e_info(e1).main is True
    v_meta = get_v_info(list(g.vs())[-1])
    assert MatchASTDict({pt: astIteratorPlaceholder(8)}) == v_meta.node_as_dict


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
    assert e1.target_vertex == a_vtx

    attr_vtx = get_v_info(e1.source_vertex)
    assert attr_vtx.v_type == Iterable[Iterable[float]]
    assert attr_vtx.node is pt
    assert MatchASTDict({pt: astIteratorPlaceholder(2)}) == attr_vtx.node_as_dict
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
    g.add_edge(a_leaf, a_vtx, info=e_info(True))

    ast_to_graph(pt, g, t_mock)

    edges = a_vtx.in_edges()
    assert len(edges) == 2

    v = list(g.vs())[-1]
    assert get_v_info(v).order == 1
    assert MatchASTDict({pt: astIteratorPlaceholder(3)}) == get_v_info(v).node_as_dict


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
    t_mock.find_broadcast_level_for_args.return_value = ((1, 1), (float, float))

    ast_to_graph(op, g, t_mock)

    assert len(g.vs()) == 3
    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[float]
    assert op_v.node is op
    assert op_v.level == 1
    assert MatchASTDict({op: astIteratorPlaceholder(1)}) == op_v.node_as_dict

    seq = op_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST(f"e1000 {sym} e2000") \
        == seq.render_ast({a: ast.Name(id='e1000'), b: ast.Name(id='e2000')})


def test_binary_op_with_parent(mocker):
    'Get iterator index right when we have a deep tree'
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.BinOp(left=a, right=b, op=ast.Add())

    g = Graph(directed=True)
    v_parent = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {ast.Constant(10): astIteratorPlaceholder(1)}))
    v1 = g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[float]], {a: astIteratorPlaceholder(2)}))
    v2 = g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[float]], {b: astIteratorPlaceholder(2)}))
    g.add_edge(v1, v_parent, info=e_info(True))
    g.add_edge(v2, v_parent, info=e_info(True))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = ((2, 2), (float, float))

    ast_to_graph(op, g, t_mock)


def test_binary_op_cross(mocker):
    'Test the binary operators'
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.BinOp(left=a, right=b, op=ast.Add())

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {b: astIteratorPlaceholder(2)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = ((1, 1), (float, float))

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


def test_binary_op_order_fixup(mocker):
    'Test the binary operators'
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.BinOp(left=a, right=b, op=ast.Add())

    g = Graph(directed=True)
    v1 = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}, order=0))
    v2 = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {b: astIteratorPlaceholder(2)}, order=0))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = ((1, 1), (float, float))

    ast_to_graph(op, g, t_mock)

    o = set([get_v_info(v1).order, get_v_info(v2).order])
    assert len(o) == 2


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
    t_mock.find_broadcast_level_for_args.return_value = ((2, 2), (float, float))

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


def test_binary_event_constant(mocker):
    'Test binary operator: Iterable[float] + Iterable[Iterable[float]] - like df.met + df.jet.pt'
    a = ast.Num('a')
    b = ast.Name('b')
    op = ast.BinOp(left=a, right=b, op=ast.Add())

    # Build the pre-existing graph. If these are from df.jets.pt, then they will already be at level 2, which
    # is where we will need to be operating too - so no change in operator.
    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[Iterable[float]], {b: astIteratorPlaceholder(2)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = ((1, 2), (float, float))

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
                                                            (Iterable[float], Iterable[Iterable[float]]))


@pytest.mark.parametrize("a, a_is_const, b, b_is_const, operator, result_string", [
                         (ast.Name(id='a'), False, ast.Num(n=10), True, ast.Add, 'a+10'),
                         (ast.Num(n=10), True, ast.Name(id='a'), False, ast.Add, '10+a'),
                         (ast.Num(n=10), True, ast.Num(n=20), True, ast.Add, '10+20'),
                         ])
def test_binary_op_constant(a: ast.AST, a_is_const: bool, b: ast.AST, b_is_const: bool, operator, result_string, mocker):
    'Test the binary operators'
    op = ast.BinOp(left=a, right=b, op=operator())

    if a_is_const:
        a_type = Constant[float]  # type: ignore
        a_dict: Dict[ast.AST, ast.AST] = {a: a}
        a_level = 0
    else:
        a_type = Iterable[float]
        a_dict = {a: astIteratorPlaceholder(1)}
        a_level = 1
    if b_is_const:
        b_type = Constant[float]  # type: ignore
        b_dict: Dict[ast.AST, ast.AST] = {b: b}
        b_level = 0
    else:
        b_type = Iterable[float]
        b_dict = {b: astIteratorPlaceholder(1)}
        b_level = 1

    g = Graph(directed=True)
    g.add_vertex(info=v_info(a_level, mocker.MagicMock(spec=sequence_predicate_base), a_type, a_dict))
    g.add_vertex(info=v_info(b_level, mocker.MagicMock(spec=sequence_predicate_base), b_type, b_dict))

    t_mock = mocker.MagicMock(spec=type_inspector)
    # Determine the type for each one
    t_mock.find_broadcast_level_for_args.return_value = ((a_level, b_level), (float, float))

    ast_to_graph(op, g, t_mock)

    assert len(g.vs()) == 3
    new_v = list(g.vs())[-1]
    op_v = get_v_info(new_v)

    if (not a_is_const) or (not b_is_const):
        assert op_v.v_type == Iterable[float]
        assert op_v.level == 1
        assert len(op_v.node_as_dict) == 1
        assert isinstance(list(op_v.node_as_dict.values())[0], astIteratorPlaceholder)
        assert len(new_v.out_edges()) == 1
    else:
        assert op_v.v_type == Constant[float]  # type: ignore
        assert op_v.level == 0
        assert len(op_v.node_as_dict) == 1
        assert not isinstance(list(op_v.node_as_dict.values())[0], astIteratorPlaceholder)
        assert len(new_v.out_edges()) == 0

    assert op_v.node is op

    seq = op_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST(result_string) \
        == seq.render_ast({})


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
    g.add_edge(n1, r, info=e_info(True))
    g.add_edge(n2, r, info=e_info(True))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = ((1, 1), (t_left, t_right))

    ast_to_graph(op, g, t_mock)

    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[t_result]


@pytest.mark.parametrize("operator, sym", [
                         (ast.Eq, '=='),
                         (ast.NotEq, '!='),
                         (ast.Gt, '>'),
                         (ast.GtE, '>='),
                         (ast.Lt, '<'),
                         (ast.LtE, '<=')
                         ])
def test_compare_op(operator, sym, mocker):
    'Test the comparison operators operators'
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.Compare(left=a, ops=[operator()], comparators=[b])

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {b: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = ((1, 1), (float, float))

    ast_to_graph(op, g, t_mock)

    assert len(g.vs()) == 3
    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[bool]
    assert op_v.node is op
    assert op_v.level == 1
    assert MatchASTDict({op: astIteratorPlaceholder(1)}) == op_v.node_as_dict

    seq = op_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST(f"e1000 {sym} e2000") \
        == seq.render_ast({a: ast.Name(id='e1000'), b: ast.Name(id='e2000')})


@pytest.mark.parametrize("l_type, r_type", [
                         (float, float),
                         (int, int),
                         (float, int),
                         (int, float),
                         ])
def test_compare_op_types(l_type, r_type, mocker):
    'Test the comparison operators operators type comparisons'
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.Compare(left=a, ops=[ast.Gt()], comparators=[b])

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[l_type], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[r_type], {b: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = ((1, 1), (l_type, r_type))

    ast_to_graph(op, g, t_mock)

    assert len(g.vs()) == 3
    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[bool]


@pytest.mark.parametrize("operator", [
                         (ast.In),
                         (ast.NotIn),
                         (ast.Is),
                         (ast.IsNot),
                         ])
def test_compare_op_bad(operator, mocker):
    'Do not support these operators'
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.Compare(left=a, ops=[operator()], comparators=[b])

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {b: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = ((1, 1), (float, float))

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(op, g, t_mock)

    assert 'unsupported' in str(e.value).lower()


def test_compare_op_types_bad(mocker):
    'Test the comparison operators operators type comparisons'
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.Compare(left=a, ops=[ast.Gt()], comparators=[b])

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[str], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {b: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = None

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(op, g, t_mock)

    assert 'unsupported' in str(e.value).lower()


def test_compare_multi_term(mocker):
    'We are not coded up to deal with multi-term compares yet'
    a = ast.Name('a')
    b = ast.Name('b')
    c = ast.Name('c')
    op = ast.Compare(left=a, ops=[ast.Gt(), ast.Lt()], comparators=[b, c])

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {b: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {c: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = ((1, 1, 1), (float, float, float))

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(op, g, t_mock)

    assert 'unsupported' in str(e.value).lower()


@pytest.mark.parametrize("operator, sym", [
                         (ast.And, "and"),
                         (ast.Or, "or"),
                         ])
def test_binary_operator(operator, sym, mocker):
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.BoolOp(op=operator(), values=[a, b])

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[bool], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[bool], {b: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = ((1, 1), (bool, bool))

    ast_to_graph(op, g, t_mock)

    assert len(g.vs()) == 3
    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[bool]
    assert op_v.node is op
    assert op_v.level == 1
    assert MatchASTDict({op: astIteratorPlaceholder(1)}) == op_v.node_as_dict

    seq = op_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST(f"e1000 {sym} e2000") \
        == seq.render_ast({a: ast.Name(id='e1000'), b: ast.Name(id='e2000')})


@pytest.mark.parametrize("operator, sym", [
                         (ast.And, "and"),
                         (ast.Or, "or"),
                         ])
def test_binary_operator_3(operator, sym, mocker):
    a = ast.Name('a')
    b = ast.Name('b')
    c = ast.Name('b')
    op = ast.BoolOp(op=operator(), values=[a, b, c])

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[bool], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[bool], {b: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[bool], {c: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = ((1, 1,), (bool, bool, bool))

    ast_to_graph(op, g, t_mock)

    assert len(g.vs()) == 4
    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[bool]
    assert op_v.node is op
    assert op_v.level == 1
    assert MatchASTDict({op: astIteratorPlaceholder(1)}) == op_v.node_as_dict

    seq = op_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST(f"e1000 {sym} e2000 {sym} e3000") \
        == seq.render_ast({a: ast.Name(id='e1000'), b: ast.Name(id='e2000'), c: ast.Name(id='e3000')})


def test_binary_operator_bad_type(mocker):
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.BoolOp(op=ast.And(), values=[a, b])

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[bool], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[str], {b: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = None

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(op, g, t_mock)

    assert "bool" in str(e.value).lower()


@pytest.mark.parametrize("operator, sym, o_type", [
                         (ast.UAdd, "+", float),
                         (ast.UAdd, "+", int),
                         (ast.USub, "-", float),
                         (ast.USub, "-", int),
                         (ast.Not, "not", bool),
                         ])
def test_unary_operator(operator, sym, o_type, mocker):
    a = ast.Name('a')
    op = ast.UnaryOp(op=operator(), operand=a)

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[bool], {a: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = ((1,), (o_type,))

    ast_to_graph(op, g, t_mock)

    assert len(g.vs()) == 2
    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[o_type]
    assert op_v.node is op
    assert op_v.level == 1
    assert MatchASTDict({op: astIteratorPlaceholder(1)}) == op_v.node_as_dict

    seq = op_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST(f"{sym} e1000") \
        == seq.render_ast({a: ast.Name(id='e1000')})


@pytest.mark.parametrize("operator, o_type, c_type", [
                         (ast.UAdd, bool, Union[float, int]),
                         (ast.USub, bool, Union[float, int]),
                         (ast.Not, int, bool),
                         (ast.Not, float, bool),
                         ])
def test_unary_perator_bad_type(operator, o_type, c_type, mocker):
    a = ast.Name('a')
    op = ast.UnaryOp(op=operator(), operand=a)

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[o_type], {a: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = None

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(op, g, t_mock)

    assert "unsupported type" in str(e.value).lower()
    t_mock.find_broadcast_level_for_args.assert_called_with((c_type,), (Iterable[o_type],))


def test_unary_invert_not_allowed(mocker):
    a = ast.Name('a')
    op = ast.UnaryOp(op=ast.Invert(), operand=a)

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[bool], {a: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = None

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(op, g, t_mock)

    assert "invert" in str(e.value).lower()


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
    t_mock.find_broadcast_level_for_args.return_value = ((1,), (float,))

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


@pytest.mark.parametrize("b_type, c_type, r_type", [
                         (int, int, int),
                         (float, float, float),
                         (int, float, float),
                         (float, int, float),
                         (str, str, str),
                         ])
def test_if_expr(b_type, c_type, r_type, mocker):
    a = ast.Name('a')
    b = ast.Name('b')
    c = ast.Name('c')
    op = ast.IfExp(test=a, body=b, orelse=c)

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[bool], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[b_type], {b: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[c_type], {c: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = ((1, 1, 1), (bool, b_type, c_type))

    ast_to_graph(op, g, t_mock)

    assert len(g.vs()) == 4
    op_v = get_v_info(list(g.vs())[-1])

    assert op_v.v_type == Iterable[r_type]
    assert op_v.node is op
    assert op_v.level == 1
    assert MatchASTDict({op: astIteratorPlaceholder(1)}) == op_v.node_as_dict

    seq = op_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST("e2000 if e1000 else e3000") \
        == seq.render_ast({a: ast.Name(id='e1000'), b: ast.Name(id='e2000'), c: ast.Name(id='e3000')})


@pytest.mark.parametrize("a_type, b_type, c_type", [
                         (bool, int, str),
                         (bool, str, float),
                         ])
def test_if_expr_bad_types(a_type, b_type, c_type, mocker):
    a = ast.Name('a')
    b = ast.Name('b')
    c = ast.Name('c')
    op = ast.IfExp(test=a, body=b, orelse=c)

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[a_type], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[b_type], {b: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[c_type], {c: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = ((1, 1, 1), (a_type, b_type, c_type))

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(op, g, t_mock)

    assert "type" in str(e.value).lower()


def test_if_expr_bad_test(mocker):
    a = ast.Name('a')
    b = ast.Name('b')
    c = ast.Name('c')
    op = ast.IfExp(test=a, body=b, orelse=c)

    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[int], {a: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {b: astIteratorPlaceholder(1)}))
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {c: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.find_broadcast_level_for_args.return_value = None

    with pytest.raises(FuncADLTablesException) as e:
        ast_to_graph(op, g, t_mock)

    assert "type" in str(e.value).lower()


def test_function_const_arg(mocker):
    'Call a function with a single argument, which is the main sequence'
    a = ast.Num(n=10)
    c = ast.Call(func=ast.Name(id='my_func'), args=[a], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    g.add_vertex(info=v_info(0, mocker.MagicMock(spec=sequence_predicate_base), Constant[float], {a: a}))  # type: ignore

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[float], float]
    t_mock.callable_type.return_value = ([float], float)
    t_mock.find_broadcast_level_for_args.return_value = ((0,), (float,))

    ast_to_graph(c, g, t_mock)

    assert len(g.vs()) == 2
    v = list(g.vs())[1]
    call_v = get_v_info(v)

    assert call_v.v_type == Constant[float]  # type: ignore
    assert call_v.node is c
    assert call_v.level == 0

    seq = call_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST("my_func(10)") \
        == seq.render_ast({})

    assert len(v.out_edges()) == 0


def test_function_two_arg(mocker):
    a = ast.Name('a')
    b = ast.Name('b')
    c = ast.Call(func=ast.Name(id='my_func'), args=[a, b], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    v_a = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))
    v_b = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {b: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[float, float], float]
    t_mock.callable_type.return_value = ([float, float], float)
    t_mock.find_broadcast_level_for_args.return_value = ((1, 1), (float, float))

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

    # Check the order
    assert get_v_info(v_a).order < get_v_info(v_b).order


def test_function_two_arg_one_const(mocker):
    a = ast.Num(n=10)
    b = ast.Name('b')
    c = ast.Call(func=ast.Name(id='my_func'), args=[a, b], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    g.add_vertex(info=v_info(0, mocker.MagicMock(spec=sequence_predicate_base), Constant[float], {a: ast.Num(n=10)}))  # type: ignore
    g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {b: astIteratorPlaceholder(1)}))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[float, float], float]
    t_mock.callable_type.return_value = ([float, float], float)
    t_mock.find_broadcast_level_for_args.return_value = ((0, 1), (float, float))

    ast_to_graph(c, g, t_mock)

    assert len(g.vs()) == 3
    call_v = get_v_info(list(g.vs())[-1])

    assert call_v.v_type == Iterable[float]
    assert call_v.node is c
    assert call_v.level == 1

    seq = call_v.sequence
    assert isinstance(seq, expression_transform)
    assert MatchAST("my_func(10, b)") \
        == seq.render_ast({})


def test_function_single_arg_level2(mocker):
    a = ast.Name('a')
    c = ast.Call(func=ast.Name(id='my_func'), args=[a], keywords=[])

    g = Graph(directed=True)
    g['info'] = g_info([])
    r = g.add_vertex(info=v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {ast.Name('b'): astIteratorPlaceholder(1)}))
    n1 = g.add_vertex(info=v_info(2, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)}))
    g.add_edge(n1, r, info=e_info(True))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[float], float]
    t_mock.callable_type.return_value = ([float], float)
    t_mock.find_broadcast_level_for_args.return_value = ((2,), (float,))

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
    t_mock.find_broadcast_level_for_args.return_value = ((2,), (float,))

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
    g.add_edge(n1, r, info=e_info(True))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.callable_signature.return_value = Callable[[float], float]
    t_mock.static_function_type.return_value = Callable[[float], float]
    t_mock.callable_type.return_value = ([float], float)
    t_mock.find_broadcast_level_for_args.return_value = ((1,), (float,))

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
    t_mock.find_broadcast_level_for_args.return_value = ((1,), (float,))

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
    g.add_edge(n1, r, info=e_info(True))

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
    g.add_edge(n1, r, info=e_info(True))
    g.add_edge(n2, r, info=e_info(True))

    t_mock = mocker.MagicMock(spec=type_inspector)
    t_mock.static_function_type.return_value = Callable[[], float]
    t_mock.callable_type.return_value = ([], float)
    t_mock.iterable_object.return_value = float
    t_mock.find_broadcast_level_for_args.return_value = ((1, 1), (float, float))

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

# TODO: Make sure df.jets.pt() works! and df.jets().pt() too.
