import ast
from typing import Dict

from igraph import Graph

from hep_tables.graph_info import e_info, get_v_info
from hep_tables.graph_linq_reducers import (find_highest_level, reduce_level,
                                            reduce_tuple_vertices,
                                            run_linear_reduction)
from hep_tables.transforms import (expression_transform, expression_tuple,
                                   sequence_downlevel)
from hep_tables.util_ast import astIteratorPlaceholder

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
    seq_met = mocker.MagicMock(spec=expression_transform)
    level_1_1 = g.add_vertex(info=mock_vinfo(mocker, node=a2_1, seq=seq_met, order=0, level=2))
    g.add_edge(level_1_1, level_0, info=e_info(True))

    reduce_level(g, 2, mock_qt)
    assert len(g.vs()) == 2
    meta = get_v_info(level_1_1)
    assert meta.node == a2_1
    a_ref = meta.node_as_dict[a2_1]
    assert isinstance(a_ref, astIteratorPlaceholder)
    assert len(a_ref.levels) == 1
    assert a_ref.levels[0] is None

    assert meta.level == 1

    s = meta.sequence
    assert isinstance(s, sequence_downlevel)
    assert s.transform is seq_met


def test_downlevel_to_zero(mocker, mock_root_sequence_transform, mock_qt):
    'Make sure a level 2 node is down-leveled correctly to level 1'
    mine, a1, root_seq = mock_root_sequence_transform
    g = Graph(directed=True)
    level_0 = g.add_vertex(info=mock_vinfo(mocker, node=a1, seq=root_seq, level=1))

    reduce_level(g, 1, mock_qt)
    assert len(g.vs()) == 1
    assert get_v_info(level_0).level == 0


def test_downlevel_one_sequence(mocker, mock_root_sequence_transform, mock_qt):
    'Make sure 2 level 2 nodes are downleveled to 2 level 1 nodes, not combined, etc.'
    mine, a1, root_seq = mock_root_sequence_transform
    g = Graph(directed=True)
    level_0 = g.add_vertex(info=mock_vinfo(mocker, node=a1, seq=root_seq, level=1))

    a2_1 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=expression_transform)
    level_1_1 = g.add_vertex(info=mock_vinfo(mocker, node=a2_1, seq=seq_met, order=0, level=2))
    g.add_edge(level_1_1, level_0, info=e_info(True))

    a3_1 = ast.Constant(10)
    seq_met_1 = mocker.MagicMock(spec=expression_transform)
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
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_1, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_2, seq=mocker.MagicMock(spec=expression_transform)))

    g.add_edge(level_2, level_1, info=e_info(True))

    reduce_tuple_vertices(g, 2, mock_qt)

    assert len(g.vs()) == 2
    assert len(get_v_info(level_1).node_as_dict[a_1].levels) == 0
    assert len(get_v_info(level_2).node_as_dict[a_2].levels) == 0


def test_reduce_vertices_simple_dependency(mocker, mock_qt):
    'Three vertices, get combined, and check meta-data'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_1, order=0, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_2, order=0, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_2, level_1, info=e_info(True))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_3, order=1, seq=mocker.MagicMock(spec=expression_transform)))
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
    assert isinstance(v_2_md.node_as_dict[a_2], astIteratorPlaceholder)
    assert v_2_md.node_as_dict[a_2].new_level == 0  # type: ignore
    assert isinstance(v_2_md.node_as_dict[a_3], astIteratorPlaceholder)
    assert v_2_md.node_as_dict[a_3].new_level == 1  # type: ignore

    assert isinstance(v_1_md.sequence, expression_transform)
    seq = v_2_md.sequence
    assert isinstance(seq, expression_tuple)
    assert len(seq.transforms) == 2

    assert v_1_md.order == 0
    assert v_2_md.order == 0


def test_reduce_vertices_separate_dependency(mocker, mock_qt):
    '''5 vertices, but they have different levesl, and despite being at same level, do not have same parents
    so do not get combined.'''
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_1, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_2, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_2, level_1, info=e_info(True))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_3, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_3, level_1, info=e_info(True))

    a_4 = ast.Constant(1)
    level_4 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_4, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_4, level_2, info=e_info(True))

    a_5 = ast.Constant(1)
    level_5 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_5, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_5, level_3, info=e_info(True))

    reduce_tuple_vertices(g, 2, mock_qt)

    assert len(g.vs()) == 5


def test_reduce_vertices_wrong_level(mocker, mock_qt):
    'Combinable vertices at level 2, but we ask for level 3 combines - so nothing happens'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_1, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_2, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_2, level_1, info=e_info(True))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_3, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_3, level_1, info=e_info(True))

    reduce_tuple_vertices(g, 3, mock_qt)

    assert len(g.vs()) == 3


def test_tuple_2levels(mocker, mock_qt):
    'Given a double level of tuples, make sure the nodes and references are combined correctly'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_1, seq=mocker.MagicMock(spec=expression_transform)))

    a_2_1 = ast.Constant(1)
    a_2_2 = ast.Constant(2)
    ast_dict: Dict[ast.AST, ast.AST] = {
        a_2_1: astIteratorPlaceholder([0]),
        a_2_2: astIteratorPlaceholder([1]),
    }
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=ast_dict, seq=mocker.MagicMock(spec=expression_transform), order=1))
    g.add_edge(level_2, level_1, info=e_info(True))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_3, seq=mocker.MagicMock(spec=expression_transform), order=0))
    g.add_edge(level_3, level_1, info=e_info(True))

    reduce_tuple_vertices(g, 2, mock_qt)

    assert len(g.vs()) == 2
    asts_dict = get_v_info(list(g.vs())[1]).node_as_dict
    assert len(asts_dict) == 3
    assert a_2_1 in asts_dict
    assert a_2_2 in asts_dict
    assert a_3 in asts_dict

    # The levels are added in reverse of how they are used, from the inner to the outter in the expression
    assert asts_dict[a_3].new_level == 0  # type: ignore
    assert asts_dict[a_2_1].new_level == 1  # type: ignore
    assert asts_dict[a_2_2].new_level == 1  # type: ignore


def test_reduce_vertices_sequential_reduction(mocker, mock_qt):
    'Two parallel paths through, when the first part of the path gets combined, the second part should too'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_1, order=0, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_2, order=1, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_2, level_1, info=e_info(True))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_3, order=2, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_3, level_1, info=e_info(True))

    a_4 = ast.Constant(1)
    level_4 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_4, order=3, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_4, level_2, info=e_info(True))

    a_5 = ast.Constant(1)
    level_5 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_5, order=4, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_5, level_3, info=e_info(True))

    reduce_tuple_vertices(g, 2, mock_qt)

    assert len(g.vs()) == 3


def test_single_to_zero(mocker, mock_qt):
    'Knock a level 2 all the way down to level 0'
    g = Graph(directed=True)
    a2_1 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=expression_transform)
    level_1_1 = g.add_vertex(info=mock_vinfo(mocker, node=a2_1, seq=seq_met, order=0, level=2))

    run_linear_reduction(g, mock_qt)

    assert len(g.vs()) == 1
    assert get_v_info(level_1_1).level == 0


def test_double_nodes_reduced_to_zero(mocker, mock_qt):
    '2 nodes, dependent on same prior node, combined and reduced in right order'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=0, node=a_1, order=0, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_2, order=1, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_2, level_1, info=e_info(True))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_3, order=2, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_3, level_1, info=e_info(True))

    run_linear_reduction(g, mock_qt)

    assert len(g.vs()) == 2
    v0, v1 = list(g.vs())
    assert get_v_info(v0).level == 0
    assert isinstance(get_v_info(v0).sequence, expression_transform)

    assert get_v_info(v1).level == 0
    seq1 = get_v_info(v1).sequence
    assert isinstance(seq1, sequence_downlevel)
    seq_tuple = seq1.transform
    assert isinstance(seq_tuple, expression_tuple)
