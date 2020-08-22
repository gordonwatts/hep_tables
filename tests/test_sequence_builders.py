import ast

from dataframe_expressions.data_frame import DataFrame
from hep_tables.exceptions import FuncADLTablesException

from func_adl import ObjectStream

from dataframe_expressions.asts import ast_DataFrame
from hep_tables import xaod_table
from typing import Callable, Iterable, List, Optional, Type

import pytest
from igraph import Graph

from hep_tables.sequence_builders import ast_to_graph
from hep_tables.transforms import astIteratorPlaceholder, root_sequence_transform, sequence_transform
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
    assert vtx['type'] == Iterable[TestEvent]
    assert vtx['node'] is a
    assert vtx['itr_depth'] == 1
    seq = vtx['seq']
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
    a_vtx = g.add_vertex(node=a, type=Iterable[TestEvent], itr_depth=1)

    ast_to_graph(pt, q_mock, g, t_mock)

    vertexes = g.vs()
    assert len(vertexes) == 2

    edges = a_vtx.in_edges()
    assert len(edges) == 1

    e1 = edges[0]
    assert e1['main_seq'] is True
    assert e1.target_vertex == a_vtx

    t_mock.attribute_type.assert_called_once()
    t_mock.attribute_type.assert_called_with(TestEvent, 'AFloat')

    attr_vtx = e1.source_vertex
    assert attr_vtx['type'] == Iterable[float]
    assert attr_vtx['node'] is pt
    assert attr_vtx['itr_depth'] == 1
    seq = attr_vtx['seq']
    assert isinstance(seq, sequence_transform)

    base = ObjectStream(ast.Name(id='dude'))
    assert MatchObjectSequence(base.Select("lambda e1000: e1000.AFloat()")) == seq.sequence(base, {})


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
    g.add_vertex(node=a, type=TestEvent)

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
    g.add_vertex(node=a, type=Iterable[TestEvent])

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
    g.add_vertex(node=a, type=Iterable[TestEvent])

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
    a_vtx = g.add_vertex(node=a, type=Iterable[Iterable[Jets]], itr_depth=1)

    ast_to_graph(pt, q_mock, g, t_mock)

    vertexes = g.vs()
    assert len(vertexes) == 2

    edges = a_vtx.in_edges()
    assert len(edges) == 1

    e1 = edges[0]
    assert e1['main_seq'] is True
    assert e1.target_vertex == a_vtx

    attr_vtx = e1.source_vertex
    assert attr_vtx['type'] == Iterable[Iterable[float]]
    assert attr_vtx['node'] is pt
    assert attr_vtx['itr_depth'] == 2
    seq = attr_vtx['seq']
    assert isinstance(seq, sequence_transform)

    base = ObjectStream(ast.Name(id='dude'))
    assert MatchObjectSequence(base.Select("lambda e1000: e1000.pt()")) == seq.sequence(base, {})


@pytest.mark.parametrize("operator", [ast.FloorDiv, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.MatMult])
def test_binary_op_unsupported(operator, mocker):
    'Test the binary operators'
    a = ast.Name('a')
    b = ast.Name('b')
    op = ast.BinOp(left=a, right=b, op=operator())

    g = Graph(directed=True)
    g.add_vertex(node=a, type=Iterable[float], itr_depth=1)
    g.add_vertex(node=b, type=Iterable[float], itr_depth=1)

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
    g.add_vertex(node=a, type=Iterable[float], itr_depth=1)
    g.add_vertex(node=b, type=Iterable[float], itr_depth=1)

    q_mock = mocker.MagicMock(spec=QueryVarTracker)
    q_mock.new_var_name.return_value = 'e1000'
    t_mock = mocker.MagicMock(spec=type_inspector)

    ast_to_graph(op, q_mock, g, t_mock)

    assert len(g.vs()) == 3
    op_v = list(g.vs())[-1]

    assert op_v['type'] == Iterable[float]
    assert op_v['node'] is op
    assert op_v['itr_depth'] == 1

    seq = op_v['seq']
    assert isinstance(seq, sequence_transform)
    base = ObjectStream(ast.Name(id='dude'))
    assert MatchObjectSequence(base.Select(f"lambda e1000: e1000 {sym} e2000")) \
        == seq.sequence(base, {a: astIteratorPlaceholder(), b: ast.Name(id='e2000')})
