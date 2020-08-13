import ast
from typing import Callable, Iterable, List

import pytest
from igraph import Graph

from hep_tables.sequence_builders import ast_to_graph
from hep_tables.transforms import sequence_transform
from hep_tables.type_info import type_inspector
from hep_tables.utils import FuncADLTablesException, QueryVarTracker


class TestEvent:
    def ListOfFloats(self) -> List[float]:
        ...

    def AFloat(self) -> float:
        ...


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

    g = Graph(directed=True)
    a_vtx = g.add_vertex(node=a, type=Iterable[TestEvent])

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
    seq = attr_vtx['seq']
    assert isinstance(seq, sequence_transform)
    assert ast.dump(seq._function) == "Call(func=Attribute(value=astIteratorPlaceholder(), attr='AFloat'), args=[], keywords={})"


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


# The same but with explicit calls.
