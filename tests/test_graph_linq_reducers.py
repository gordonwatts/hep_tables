import ast
from hep_tables.graph_info import e_info, get_v_info

from igraph import Graph

from hep_tables.graph_linq_reducers import (find_highest_level, reduce_level,
                                            reduce_tuple_vertices)
from hep_tables.transforms import (astIteratorPlaceholder, sequence_downlevel, sequence_transform,
                                   sequence_tuple)
from .conftest import MatchAST, mock_vinfo


def test_level_one_node(mocker):
    g = Graph(directed=True)
    g.add_vertex(info=mock_vinfo(mocker, level=1))
    assert find_highest_level(g) == 1


def test_level_multiple_skip(mocker):
    g = Graph(directed=True)
    g.add_vertex(info=mock_vinfo(mocker, level=1))
    g.add_vertex(info=mock_vinfo(mocker, level=3))
    g.add_vertex(info=mock_vinfo(mocker, level=1))
    assert find_highest_level(g) == 3


def test_downlevel_one(mocker, mock_root_sequence_transform, mock_qt):
    'Make sure a level 2 node is down-leveled correctly to level 1'
    mine, a1, root_seq = mock_root_sequence_transform
    g = Graph(directed=True)
    level_0 = g.add_vertex(info=mock_vinfo(mocker, node=a1, seq=root_seq, level=1))

    a2_1 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=sequence_transform)
    level_1_1 = g.add_vertex(info=mock_vinfo(mocker, node=a2_1, seq=seq_met, order=0, level=2))
    g.add_edge(level_1_1, level_0, info=e_info(True))

    reduce_level(g, 2, mock_qt)
    assert len(g.vs()) == 2
    meta = get_v_info(level_1_1)
    assert meta.node == a2_1
    assert meta.level == 1

    s = meta.sequence
    assert isinstance(s, sequence_downlevel)
    assert s.transform is seq_met


def test_downlevel_one_sequence(mocker, mock_root_sequence_transform, mock_qt):
    'Make sure 2 level 2 nodes are downleveled to 2 level 1 nodes, not combined, etc.'
    mine, a1, root_seq = mock_root_sequence_transform
    g = Graph(directed=True)
    level_0 = g.add_vertex(info=mock_vinfo(mocker, node=a1, seq=root_seq, level=1))

    a2_1 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=sequence_transform)
    level_1_1 = g.add_vertex(info=mock_vinfo(mocker, node=a2_1, seq=seq_met, order=0, level=2))
    g.add_edge(level_1_1, level_0, info=e_info(True))

    a3_1 = ast.Constant(10)
    seq_met_1 = mocker.MagicMock(spec=sequence_transform)
    level_2_1 = g.add_vertex(info=mock_vinfo(mocker, node=a3_1, seq=seq_met_1, order=0, level=2))
    g.add_edge(level_2_1, level_1_1, info=e_info(True))

    reduce_level(g, 2, mock_qt)
    assert len(g.vs()) == 3
    assert get_v_info(level_1_1).level == 1
    assert get_v_info(level_2_1).level == 1


def test_reduce_vertices_separate_steps(mocker, mock_qt):
    'Two vertices in different steps, same level, do not get combined'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_1, seq=mocker.MagicMock(spec=sequence_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_2, seq=mocker.MagicMock(spec=sequence_transform)))

    g.add_edge(level_2, level_1, info=e_info(True))

    reduce_tuple_vertices(g, 2, mock_qt)

    assert len(g.vs()) == 2


def test_reduce_vertices_simple_dependency(mocker, mock_qt):
    'Three vertices, get combined, and check meta-data'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_1, order=0, seq=mocker.MagicMock(spec=sequence_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_2, order=0, seq=mocker.MagicMock(spec=sequence_transform)))
    g.add_edge(level_2, level_1, info=e_info(True))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_3, order=1, seq=mocker.MagicMock(spec=sequence_transform)))
    g.add_edge(level_3, level_1, info=e_info(True))

    reduce_tuple_vertices(g, 2, mock_qt)

    assert len(g.vs()) == 2  # Number of vertices
    assert len(g.es()) == 1  # Number of edges

    v_1, v_2 = list(g.vs())
    assert len(v_1.neighbors(mode='out')) == 0
    assert len(v_1.neighbors(mode='in')) == 1
    assert len(v_2.neighbors(mode='out')) == 1
    assert len(v_2.neighbors(mode='in')) == 0

    v_1_md = get_v_info(v_1)
    v_2_md = get_v_info(v_2)
    assert v_1_md.level == 2
    assert v_2_md.level == 2

    assert v_1_md.node is a_1
    assert len(v_2_md.node_as_dict) == 2
    assert MatchAST(ast.Subscript(value=astIteratorPlaceholder(), slice=ast.Index(value=0))) == v_2_md.node_as_dict[a_2]
    assert MatchAST(ast.Subscript(value=astIteratorPlaceholder(), slice=ast.Index(value=1))) == v_2_md.node_as_dict[a_3]

    assert isinstance(v_1_md.sequence, sequence_transform)
    seq = v_2_md.sequence
    assert isinstance(seq, sequence_tuple)
    assert len(seq.transforms) == 2

    assert v_1_md.order == 0
    assert v_2_md.order == 0


def test_reduce_vertices_separate_dependency(mocker, mock_qt):
    '''5 vertices, but they have different levesl, and despite being at same level, do not have same parents
    so do not get combined.'''
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_1, seq=mocker.MagicMock(spec=sequence_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_2, seq=mocker.MagicMock(spec=sequence_transform)))
    g.add_edge(level_2, level_1, info=e_info(True))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_3, seq=mocker.MagicMock(spec=sequence_transform)))
    g.add_edge(level_3, level_1, info=e_info(True))

    a_4 = ast.Constant(1)
    level_4 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_4, seq=mocker.MagicMock(spec=sequence_transform)))
    g.add_edge(level_4, level_2, info=e_info(True))

    a_5 = ast.Constant(1)
    level_5 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_5, seq=mocker.MagicMock(spec=sequence_transform)))
    g.add_edge(level_5, level_3, info=e_info(True))

    reduce_tuple_vertices(g, 2, mock_qt)

    assert len(g.vs()) == 5


def test_reduce_vertices_wrong_level(mocker, mock_qt):
    'Combinable vertices at level 2, but we ask for level 3 combines - so nothing happens'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_1, seq=mocker.MagicMock(spec=sequence_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_2, seq=mocker.MagicMock(spec=sequence_transform)))
    g.add_edge(level_2, level_1, info=e_info(True))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_3, seq=mocker.MagicMock(spec=sequence_transform)))
    g.add_edge(level_3, level_1, info=e_info(True))

    reduce_tuple_vertices(g, 3, mock_qt)

    assert len(g.vs()) == 3


def test_reduce_vertices_sequential_reduction(mocker, mock_qt):
    'Two parallel paths through, when the first part of the path gets combined, the second part should too'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_1, order=0, seq=mocker.MagicMock(spec=sequence_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_2, order=1, seq=mocker.MagicMock(spec=sequence_transform)))
    g.add_edge(level_2, level_1, info=e_info(True))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_3, order=2, seq=mocker.MagicMock(spec=sequence_transform)))
    g.add_edge(level_3, level_1, info=e_info(True))

    a_4 = ast.Constant(1)
    level_4 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_4, order=3, seq=mocker.MagicMock(spec=sequence_transform)))
    g.add_edge(level_4, level_2, info=e_info(True))

    a_5 = ast.Constant(1)
    level_5 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_5, order=4, seq=mocker.MagicMock(spec=sequence_transform)))
    g.add_edge(level_5, level_3, info=e_info(True))

    reduce_tuple_vertices(g, 2, mock_qt)

    assert len(g.vs()) == 3
