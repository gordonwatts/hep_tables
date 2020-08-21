import ast

from igraph import Graph

from hep_tables.graph_linq_reducers import (find_highest_level, reduce_level,
                                            reduce_tuple_vertices)
from hep_tables.transforms import (sequence_downlevel, sequence_transform,
                                   sequence_tuple)


def test_level_one_node():
    g = Graph(directed=True)
    g.add_vertex(itr_depth=1)
    assert find_highest_level(g) == 1


def test_level_multiple_skip():
    g = Graph(directed=True)
    g.add_vertex(itr_depth=1)
    g.add_vertex(itr_depth=3)
    g.add_vertex(itr_depth=1)
    assert find_highest_level(g) == 3


def test_downlevel_one(mocker, mock_root_sequence_transform, mock_qt):
    'Make sure a level 2 node is down-leveled correctly to level 1'
    mine, a1, root_seq = mock_root_sequence_transform
    g = Graph(directed=True)
    level_0 = g.add_vertex(node=a1, seq=root_seq, itr_depth=1)

    a2_1 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=sequence_transform)
    level_1_1 = g.add_vertex(node=a2_1, seq=seq_met, order=0, itr_depth=2)
    g.add_edge(level_1_1, level_0, main_seq=True)

    reduce_level(g, 2, mock_qt)
    assert len(g.vs()) == 2
    assert level_1_1['itr_depth'] == 1
    assert level_1_1['node'] == a2_1

    s = level_1_1['seq']
    assert isinstance(s, sequence_downlevel)
    assert s.transform == seq_met


def test_downlevel_one_sequence(mocker, mock_root_sequence_transform, mock_qt):
    'Make sure 2 level 2 nodes are downleveled to 2 level 1 nodes, not combined, etc.'
    mine, a1, root_seq = mock_root_sequence_transform
    g = Graph(directed=True)
    level_0 = g.add_vertex(node=a1, seq=root_seq, itr_depth=1)

    a2_1 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=sequence_transform)
    level_1_1 = g.add_vertex(node=a2_1, seq=seq_met, order=0, itr_depth=2)
    g.add_edge(level_1_1, level_0, main_seq=True)

    a3_1 = ast.Constant(10)
    seq_met_1 = mocker.MagicMock(spec=sequence_transform)
    level_2_1 = g.add_vertex(node=a3_1, seq=seq_met_1, order=0, itr_depth=2)
    g.add_edge(level_2_1, level_1_1, main_seq=True)

    reduce_level(g, 2, mock_qt)
    assert len(g.vs()) == 3
    assert level_1_1['itr_depth'] == 1
    assert level_2_1['itr_depth'] == 1


def test_reduce_vertices_separate_steps(mocker):
    'Two vertices in different steps, same level, do not get combined'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(itr_depth=2, node=a_1, seq=mocker.MagicMock(spec=sequence_transform))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(itr_depth=2, node=a_2, seq=mocker.MagicMock(spec=sequence_transform))

    g.add_edge(level_2, level_1, main_seq=True)

    reduce_tuple_vertices(g, 2)

    assert len(g.vs()) == 2


def test_reduce_vertices_simple_dependency(mocker):
    'Three vertices, get combined, and check meta-data'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(itr_depth=2, node=a_1, order=0, seq=mocker.MagicMock(spec=sequence_transform))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(itr_depth=2, node=a_2, order=0, seq=mocker.MagicMock(spec=sequence_transform))
    g.add_edge(level_2, level_1, main_seq=True)

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(itr_depth=2, node=a_3, order=1, seq=mocker.MagicMock(spec=sequence_transform))
    g.add_edge(level_3, level_1, main_seq=True)

    reduce_tuple_vertices(g, 2)

    assert len(g.vs()) == 2  # Number of vertices
    assert len(g.es()) == 1  # Number of edges

    v_1, v_2 = list(g.vs())
    assert len(v_1.neighbors(mode='out')) == 0
    assert len(v_1.neighbors(mode='in')) == 1
    assert len(v_2.neighbors(mode='out')) == 1
    assert len(v_2.neighbors(mode='in')) == 0

    assert v_1['itr_depth'] == 2
    assert v_2['itr_depth'] == 2

    assert v_1['node'] is a_1
    assert len(v_2['node']) == 2
    assert v_2['node'][0] is a_2
    assert v_2['node'][1] is a_3

    assert isinstance(v_1['seq'], sequence_transform)
    seq = v_2['seq']
    assert isinstance(seq, sequence_tuple)
    assert len(seq.transforms) == 2

    assert v_1['order'] == 0
    assert v_2['order'] == 0


def test_reduce_vertices_separate_dependency(mocker):
    '''5 vertices, but they have different levesl, and despite being at same level, do not have same parents
    so do not get combined.'''
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(itr_depth=1, node=a_1, seq=mocker.MagicMock(spec=sequence_transform))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(itr_depth=1, node=a_2, seq=mocker.MagicMock(spec=sequence_transform))
    g.add_edge(level_2, level_1, main_seq=True)

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(itr_depth=2, node=a_3, seq=mocker.MagicMock(spec=sequence_transform))
    g.add_edge(level_3, level_1, main_seq=True)

    a_4 = ast.Constant(1)
    level_4 = g.add_vertex(itr_depth=2, node=a_4, seq=mocker.MagicMock(spec=sequence_transform))
    g.add_edge(level_4, level_2, main_seq=True)

    a_5 = ast.Constant(1)
    level_5 = g.add_vertex(itr_depth=2, node=a_5, seq=mocker.MagicMock(spec=sequence_transform))
    g.add_edge(level_5, level_3, main_seq=True)

    reduce_tuple_vertices(g, 2)

    assert len(g.vs()) == 5


def test_reduce_vertices_wrong_level(mocker):
    'Combinable vertices at level 2, but we ask for level 3 combines - so nothing happens'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(itr_depth=2, node=a_1, seq=mocker.MagicMock(spec=sequence_transform))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(itr_depth=2, node=a_2, seq=mocker.MagicMock(spec=sequence_transform))
    g.add_edge(level_2, level_1, main_seq=True)

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(itr_depth=2, node=a_3, seq=mocker.MagicMock(spec=sequence_transform))
    g.add_edge(level_3, level_1, main_seq=True)

    reduce_tuple_vertices(g, 3)

    assert len(g.vs()) == 3


def test_reduce_vertices_sequential_reduction(mocker):
    'Two parallel paths through, when the first part of the path gets combined, the second part should too'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(itr_depth=2, node=a_1, order=0, seq=mocker.MagicMock(spec=sequence_transform))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(itr_depth=2, node=a_2, order=1, seq=mocker.MagicMock(spec=sequence_transform))
    g.add_edge(level_2, level_1, main_seq=True)

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(itr_depth=2, node=a_3, order=2, seq=mocker.MagicMock(spec=sequence_transform))
    g.add_edge(level_3, level_1, main_seq=True)

    a_4 = ast.Constant(1)
    level_4 = g.add_vertex(itr_depth=2, node=a_4, order=3, seq=mocker.MagicMock(spec=sequence_transform))
    g.add_edge(level_4, level_2, main_seq=True)

    a_5 = ast.Constant(1)
    level_5 = g.add_vertex(itr_depth=2, node=a_5, order=4, seq=mocker.MagicMock(spec=sequence_transform))
    g.add_edge(level_5, level_3, main_seq=True)

    reduce_tuple_vertices(g, 2)

    assert len(g.vs()) == 3
