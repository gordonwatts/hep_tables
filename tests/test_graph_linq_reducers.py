import ast
from typing import Dict

from igraph import Graph, Vertex  # type: ignore

from hep_tables.graph_info import e_info, g_info, get_e_info, get_v_info
from hep_tables.graph_linq_reducers import (find_highest_level, reduce_iterator_chaining, reduce_level,
                                            reduce_tuple_vertices,
                                            run_linear_reduction)
from hep_tables.transforms import (expression_transform, expression_tuple,
                                   sequence_downlevel)
from hep_tables.util_ast import astIteratorPlaceholder

from .conftest import MatchAST, MatchASTDict, mock_vinfo, parse_ast_string


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
    'Make sure a level 2 node is down-leveled correctly to level 1, same itr index'
    mine, a1, root_seq = mock_root_sequence_transform
    g = Graph(directed=True)
    level_0 = g.add_vertex(info=mock_vinfo(mocker, node=a1, seq=root_seq, level=1))

    a2_1 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=expression_transform)
    level_1_1 = g.add_vertex(info=mock_vinfo(mocker, node=a2_1, seq=seq_met, order=0, level=2))
    g.add_edge(level_1_1, level_0, info=e_info(True, 1))

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
    assert s.sequence_ast is a1
    assert s.iterator_idx == 1


def test_downlevel_different_iterators(mocker, mock_root_sequence_transform, mock_qt):
    'Make sure a level 2 node is down-leveled correctly to level 1'
    mine, a1, root_seq = mock_root_sequence_transform
    g = Graph(directed=True)
    level_0 = g.add_vertex(info=mock_vinfo(mocker, node={a1: astIteratorPlaceholder(1)},
                                           seq=root_seq, level=1))

    a2_1 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=expression_transform)
    level_1_1 = g.add_vertex(info=mock_vinfo(mocker, node={a2_1: astIteratorPlaceholder(2)},
                             seq=seq_met, order=0, level=2))
    g.add_edge(level_1_1, level_0, info=e_info(True, 1))

    reduce_level(g, 2, mock_qt)

    meta = get_v_info(level_1_1)

    a_ref = meta.node_as_dict[a2_1]
    assert isinstance(a_ref, astIteratorPlaceholder)
    assert len(a_ref.levels) == 1
    assert a_ref.levels[0] is None

    s = meta.sequence
    assert isinstance(s, sequence_downlevel)
    assert s.iterator_idx == 1


def test_main_seq_seen_on_2node_reduction(mocker, mock_qt):
    'One note is dependent on two nodes. Make sure the main seq ast is marked in the downlevel node'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=0, node=a_1, order=0, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(2)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=0, node=a_2, order=0, seq=mocker.MagicMock(spec=expression_transform)))

    a_3 = ast.Constant(3)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_3, order=0, seq=mocker.MagicMock(spec=expression_transform)))

    g.add_edge(level_3, level_2, info=e_info(True, 1))
    g.add_edge(level_3, level_1, info=e_info(False, 1))

    reduce_level(g, 2, mock_qt)

    v = list(g.vs())[-1]
    v_info = get_v_info(v)

    s = v_info.sequence
    assert isinstance(s, sequence_downlevel)
    assert s.sequence_ast is a_2


def test_downlevel_to_zero(mocker, mock_root_sequence_transform, mock_qt):
    'Make sure a level 2 node is down-leveled correctly to level 1'
    mine, a1, root_seq = mock_root_sequence_transform
    g = Graph(directed=True)
    level_0 = g.add_vertex(info=mock_vinfo(mocker, node=a1, seq=root_seq, level=0))

    level_1 = g.add_vertex(info=mock_vinfo(mocker, node=a1, seq=root_seq, level=1))
    g.add_edge(level_1, level_0, info=e_info(True, 1))

    reduce_level(g, 1, mock_qt)
    assert len(g.vs()) == 2
    assert get_v_info(level_1).level == 0


def test_downlevel_one_sequence(mocker, mock_root_sequence_transform, mock_qt):
    'Make sure 2 level 2 nodes are downleveled to 2 level 1 nodes, not combined, etc.'
    mine, a1, root_seq = mock_root_sequence_transform
    g = Graph(directed=True)
    level_0 = g.add_vertex(info=mock_vinfo(mocker, node=a1, seq=root_seq, level=1))

    a2_1 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=expression_transform)
    level_1_1 = g.add_vertex(info=mock_vinfo(mocker, node=a2_1, seq=seq_met, order=0, level=2))
    g.add_edge(level_1_1, level_0, info=e_info(True, 1))

    a3_1 = ast.Constant(10)
    seq_met_1 = mocker.MagicMock(spec=expression_transform)
    level_2_1 = g.add_vertex(info=mock_vinfo(mocker, node=a3_1, seq=seq_met_1, order=0, level=2))
    g.add_edge(level_2_1, level_1_1, info=e_info(True, 1))

    reduce_level(g, 2, mock_qt)
    assert len(g.vs()) == 3
    assert get_v_info(level_1_1).level == 1
    assert get_v_info(level_2_1).level == 1


def test_downlevel_with_combined_node(mocker, mock_qt):
    '''Properly deal with reducing a level when the one we are reducing is represented
    by multiple ast's. Typically - after something has gone through a reduction by tuple, for example.
    '''
    g = Graph(directed=True)
    a1_1 = ast.Num(n=1)
    a1_2 = ast.Num(n=2)
    node_0 = g.add_vertex(info=mock_vinfo(mocker, node={a1_1: astIteratorPlaceholder(1), a1_2: astIteratorPlaceholder(1)},
                                          seq=mocker.MagicMock(spec=expression_transform), order=0, level=0))

    a2 = ast.Num(n=2)
    node_1 = g.add_vertex(info=mock_vinfo(mocker, node=a2,
                                          seq=mocker.MagicMock(spec=expression_transform), order=0, level=2))
    g.add_edge(node_1, node_0, info=e_info(True, 1))

    reduce_level(g, 2, mock_qt)

    assert len(g.vs()) == 2
    info = get_v_info(node_1)
    assert info.level == 1
    assert len(info.node_as_dict) == 1


def test_reduce_vertices_separate_steps(mocker, mock_qt):
    'Two vertices in different steps, same level, do not get combined'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_1, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_2, seq=mocker.MagicMock(spec=expression_transform)))

    g.add_edge(level_2, level_1, info=e_info(True, 1))

    reduce_tuple_vertices(g, 2, mock_qt)

    assert len(g.vs()) == 2
    assert len(get_v_info(level_1).node_as_dict[a_1].levels) == 0  # type: ignore
    assert len(get_v_info(level_2).node_as_dict[a_2].levels) == 0  # type: ignore


def test_reduce_vertices_simple_dependency(mocker, mock_qt):
    '''Three vertices, get combined, all on same iterator at same level. This is
    similar to Jet object -> jet.p1 + jet.p2 - implied loop, same iterator'''
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=2, node={a_1: astIteratorPlaceholder(1)},
                           order=0, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node={a_2: astIteratorPlaceholder(1)},
                           order=0, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_2, level_1, info=e_info(True, 1))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node={a_3: astIteratorPlaceholder(1)},
                           order=1, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_3, level_1, info=e_info(True, 1))

    reduce_tuple_vertices(g, 2, mock_qt)

    assert len(g.vs()) == 2  # Number of vertices
    assert len(g.es()) == 1  # Number of edges

    v_1, v_2 = list(g.vs())
    assert len(v_1.neighbors(mode='out')) == 0
    assert len(v_1.neighbors(mode='in')) == 1
    assert len(v_2.neighbors(mode='out')) == 1
    assert len(v_2.neighbors(mode='in')) == 0

    edge_info = get_e_info(v_2.out_edges()[0])
    assert edge_info.main
    assert edge_info.itr_idx == 1

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
    assert v_2_md.order == 1


def test_reduce_vertices_simple_change_iterator_index(mocker, mock_qt):
    '''Check that the `node_as_dict` gets properly updated with levels when
    we do down a level when the iterator number changes. This is making sure that
    the right iterator gets updated.

    Iterable[Jets] -> jet.pt1 + jet.pt2 at level + 1.
    '''
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=1, node={a_1: astIteratorPlaceholder(1)},
                           order=0, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node={a_2: astIteratorPlaceholder(2)},
                           order=0, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_2, level_1, info=e_info(True, 1))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node={a_3: astIteratorPlaceholder(2)},
                           order=1, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_3, level_1, info=e_info(True, 1))

    reduce_tuple_vertices(g, 2, mock_qt)

    v_1, v_2 = list(g.vs())

    edge_info = get_e_info(v_2.out_edges()[0])
    assert edge_info.main
    assert edge_info.itr_idx == 1

    v_2_md = get_v_info(v_2)
    assert v_2_md.level == 2

    assert len(v_2_md.node_as_dict) == 2
    assert isinstance(v_2_md.node_as_dict[a_2], astIteratorPlaceholder)
    assert v_2_md.node_as_dict[a_2].new_level == 0  # type: ignore
    assert isinstance(v_2_md.node_as_dict[a_3], astIteratorPlaceholder)
    assert v_2_md.node_as_dict[a_3].new_level == 1  # type: ignore


def test_reduce_vertices_itr_idx_check(mocker, mock_qt):
    '''Three vertices, get combined, all on same iterator at same level. This is
    similar to Jet object -> jet.p1 + jet.p2 - implied loop, use
    a different iterator number'''
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=2, node={a_1: astIteratorPlaceholder(2)},
                           order=0, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node={a_2: astIteratorPlaceholder(2)},
                           order=0, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_2, level_1, info=e_info(True, 2))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node={a_3: astIteratorPlaceholder(2)},
                           order=1, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_3, level_1, info=e_info(True, 2))

    reduce_tuple_vertices(g, 2, mock_qt)

    assert len(g.vs()) == 2  # Number of vertices
    assert len(g.es()) == 1  # Number of edges

    v_1, v_2 = list(g.vs())
    assert len(v_1.neighbors(mode='out')) == 0
    assert len(v_1.neighbors(mode='in')) == 1
    assert len(v_2.neighbors(mode='out')) == 1
    assert len(v_2.neighbors(mode='in')) == 0

    edge_info = get_e_info(v_2.out_edges()[0])
    assert edge_info.main
    assert edge_info.itr_idx == 2


def test_reduce_tuple_downstream_edges_updated(mocker, mock_qt):
    '''Two vertices get combined. Make sure dependent that combines has edges updated correctly.
    This is like having jets => jets.pt1, jets.pt2, func(jets.pt1 (array), jets.pt2 (array))'''
    g = Graph(directed=True)
    a_1 = ast.Num(n=1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_1, order=0, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Num(n=2)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_2, order=0, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_2, level_1, info=e_info(True, 1))

    a_3 = ast.Num(n=3)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_3, order=1, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_3, level_1, info=e_info(True, 1))

    a_4 = ast.Num(n=4)
    level_4 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_4, order=1, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_4, level_3, info=e_info(True, 2))
    g.add_edge(level_4, level_2, info=e_info(True, 2))

    reduce_tuple_vertices(g, 2, mock_qt)

    assert len(g.vs()) == 3  # Number of vertices
    assert len(g.es()) == 2  # Number of edges

    # All edges should be main edges
    for e in g.es():
        assert get_e_info(e).main


def test_reduce_vertices_separate_dependency(mocker, mock_qt):
    '''5 vertices, but they have different levesl, and despite being at same level, do not have same parents
    so do not get combined.'''
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_1, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_2, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_2, level_1, info=e_info(True, 1))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_3, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_3, level_1, info=e_info(True, 1))

    a_4 = ast.Constant(1)
    level_4 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_4, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_4, level_2, info=e_info(True, 1))

    a_5 = ast.Constant(1)
    level_5 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_5, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_5, level_3, info=e_info(True, 1))

    reduce_tuple_vertices(g, 2, mock_qt)

    assert len(g.vs()) == 5


def test_reduce_vertices_wrong_level(mocker, mock_qt):
    'Combinable vertices at level 2, but we ask for level 3 combines - so nothing happens'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_1, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_2, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_2, level_1, info=e_info(True, 1))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_3, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_3, level_1, info=e_info(True, 1))

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
        a_2_1: astIteratorPlaceholder(1, [0]),
        a_2_2: astIteratorPlaceholder(1, [1]),
    }
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=ast_dict, seq=mocker.MagicMock(spec=expression_transform), order=1))
    g.add_edge(level_2, level_1, info=e_info(True, 1))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_3, seq=mocker.MagicMock(spec=expression_transform), order=0))
    g.add_edge(level_3, level_1, info=e_info(True, 1))

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
    a_1 = ast.Num(n=1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_1, order=0, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Num(n=2)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_2, order=1, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_2, level_1, info=e_info(True, 1))

    a_3 = ast.Num(n=3)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_3, order=2, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_3, level_1, info=e_info(True, 1))

    a_4 = ast.Num(n=4)
    level_4 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_4, order=3, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_4, level_2, info=e_info(True, 1))

    a_5 = ast.Num(n=5)
    level_5 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_5, order=4, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_5, level_3, info=e_info(True, 1))

    reduce_tuple_vertices(g, 2, mock_qt)

    assert len(g.vs()) == 3
    assert len(g.es()) == 2
    for e in g.es():
        assert get_e_info(e).main


def test_two_parents_tuple_combined(mocker, mock_qt):
    'Have two parents for a item, and then combine them - make sure main_seq is good'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    node_1 = g.add_vertex(info=mock_vinfo(mocker, level=0, node=a_1, order=0, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(1)
    node_2 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_2, order=0, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(node_2, node_1, info=e_info(True, 1))

    a_3 = ast.Constant(1)
    node_3 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_3, order=1, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(node_3, node_1, info=e_info(True, 1))

    a_4 = ast.Constant(1)
    node_4 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_4, order=2, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(node_4, node_2, info=e_info(False, 1))
    g.add_edge(node_4, node_3, info=e_info(True, 1))

    a_5 = ast.Constant(1)
    node_5 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_5, order=3, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(node_5, node_2, info=e_info(False, 1))
    g.add_edge(node_5, node_3, info=e_info(True, 1))

    a_6 = ast.Constant(6)
    node_6 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_6, order=4, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(node_6, node_4, info=e_info(True, 1))
    g.add_edge(node_6, node_5, info=e_info(False, 1))

    reduce_tuple_vertices(g, 2, mock_qt)

    # Check that the proper node gets listed as main_seq
    def find_node_with_ast(g: Graph, find_a: ast.AST):
        def has_ast(a: ast.AST, v: Vertex):
            v_info = get_v_info(v)
            return a in v_info.node_as_dict

        found = [v for v in g.vs() if has_ast(find_a, v)]
        assert len(found) == 1
        return found[0]

    assert len(g.vs()) == 5

    # Check the combined node has proper edges to prior guys
    v = find_node_with_ast(g, a_5)
    edges = list(v.out_edges())
    assert len(edges) == 2
    is_main = [get_e_info(e).main for e in edges]
    assert not (is_main[0] and is_main[1])
    assert is_main[0] or is_main[1]

    e_main = 0 if is_main[0] else 1
    e_other = 1 if is_main[0] else 0
    assert edges[e_main].target_vertex == node_3
    assert edges[e_other].target_vertex == node_2

    # Check the downstream vertex that is now pointing to a single item.
    v = find_node_with_ast(g, a_6)
    edges = list(v.out_edges())
    assert len(edges) == 1
    assert get_e_info(edges[0]).main


def test_two_parents_tuple_combined_reorder(mocker, mock_qt):
    'Have two parents for a item, and then combine them - make sure main_seq is good'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    node_1 = g.add_vertex(info=mock_vinfo(mocker, level=0, node=a_1, order=0, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(1)
    node_2 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_2, order=1, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(node_2, node_1, info=e_info(True, 1))

    a_3 = ast.Constant(1)
    node_3 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_3, order=2, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(node_3, node_1, info=e_info(True, 1))

    a_4 = ast.Constant(1)
    node_4 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_4, order=3, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(node_4, node_2, info=e_info(True, 1))
    g.add_edge(node_4, node_3, info=e_info(False, 1))

    a_5 = ast.Constant(1)
    node_5 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a_5, order=4, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(node_5, node_2, info=e_info(True, 1))
    g.add_edge(node_5, node_3, info=e_info(False, 1))

    a_6 = ast.Constant(6)
    node_6 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_6, order=5, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(node_6, node_4, info=e_info(True, 1))
    g.add_edge(node_6, node_5, info=e_info(False, 1))

    reduce_tuple_vertices(g, 2, mock_qt)

    # Check that the proper node gets listed as main_seq
    def find_node_with_ast(g: Graph, find_a: ast.AST):
        def has_ast(a: ast.AST, v: Vertex):
            v_info = get_v_info(v)
            return a in v_info.node_as_dict

        found = [v for v in g.vs() if has_ast(find_a, v)]
        assert len(found) == 1
        return found[0]

    assert len(g.vs()) == 5

    # Check the combined node has proper edges to prior guys
    v = find_node_with_ast(g, a_5)
    edges = list(v.out_edges())
    assert len(edges) == 2
    is_main = [get_e_info(e).main for e in edges]
    assert not (is_main[0] and is_main[1])
    assert is_main[0] or is_main[1]

    assert (is_main[0] and not is_main[1]) or (is_main[1] and not is_main[0])
    e_main = 0 if is_main[0] else 1
    e_other = 1 if is_main[0] else 0
    assert edges[e_main].target_vertex == node_2
    assert edges[e_other].target_vertex == node_3

    # Check the downstream vertex that is now pointing to a single item.
    v = find_node_with_ast(g, a_6)
    edges = list(v.out_edges())
    assert len(edges) == 1
    assert get_e_info(edges[0]).main


def test_two_iterators_replacement(mocker, mock_qt):
    'Two parent nodes, and different iteration sequences. Make sure they are correctly combined.'
    g = Graph(directed=True)

    # Now, the two nodes that have different iterator sequences
    a2 = ast.Num(n=2)
    node2 = g.add_vertex(info=mock_vinfo(mocker, level=1, node={a2: astIteratorPlaceholder(1, [0])}, order=1, seq=mocker.MagicMock(spec=expression_transform)))

    a3 = ast.Num(n=3)
    node3 = g.add_vertex(info=mock_vinfo(mocker, level=1, node={a3: astIteratorPlaceholder(2, [1])}, order=1, seq=mocker.MagicMock(spec=expression_transform)))

    # Next, the node that uses both of them.
    a4 = ast.Num(n=4)
    seq4 = mocker.MagicMock(spec=expression_transform)
    ast_name_dict: Dict[str, ast.AST] = {"a2": a2, "a3": a3}
    seq4.render_ast.return_value = parse_ast_string("a2 + e1000", ast_name_dict)
    node4 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a4, order=1, seq=seq4))
    g.add_edge(node4, node2, info=e_info(True, 1))
    g.add_edge(node4, node3, info=e_info(False, 2))

    # Now do the iterator reduction. It should hit this node.
    reduce_iterator_chaining(g, level=2, qt=mock_qt)

    # Basics of the node topology should not have changed
    assert len(g.vs()) == 3
    assert len(g.es()) == 2

    assert node2 in node4.neighbors(mode='out')
    assert node3 in node4.neighbors(mode='out')

    # The iterators should now both be marked as 1 (since that is main sequence)
    for e in g.es():
        assert get_e_info(e).itr_idx == 1

    assert MatchAST("astIteratorPlaceholder(1, [1])") == get_v_info(node3).node_as_dict[a3]

    # However, the sequence should have changed into some thing that is a second derivation.
    new_seq4 = get_v_info(node4).sequence
    assert MatchAST("Select(a3, lambda e1000: a2 + e1000)", ast_name_dict) == new_seq4.render_ast({})
    seq4.render_ast.assert_called_with(MatchASTDict({a3: ast.Name(id='e1000')}))


def test_two_iterators_same_iterator(mocker, mock_qt):
    'If we do not have two different iterators, then it should do no chaining'
    g = Graph(directed=True)

    # Now, the two nodes that have different iterator sequences
    a2 = ast.Num(n=2)
    node2 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a2, order=1, seq=mocker.MagicMock(spec=expression_transform)))

    a3 = ast.Num(n=3)
    node3 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a3, order=1, seq=mocker.MagicMock(spec=expression_transform)))

    # Next, the node that uses both of them.
    a4 = ast.Num(n=4)
    seq4 = mocker.MagicMock(spec=expression_transform)
    node4 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a4, order=1, seq=seq4))
    g.add_edge(node4, node2, info=e_info(True, 1))
    g.add_edge(node4, node3, info=e_info(False, 1))

    # Now do the iterator reduction. It should hit this node.
    reduce_iterator_chaining(g, level=2, qt=mock_qt)

    # However, the sequence should have changed into some thing that is a second derivation.
    assert get_v_info(node4).sequence is seq4


def test_two_same_iterators_from_one_node(mocker, mock_qt):
    '''A node has two dependent iterators, but they are the same iterator
    If we do not have two different iterators, then it should do nothing'''
    g = Graph(directed=True)

    # Now, the three nodes, one uses one iterator sequence, the other two use a second one.
    a2 = ast.Num(n=2)
    node2 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a2, order=1, seq=mocker.MagicMock(spec=expression_transform)))

    a3 = ast.Num(n=3)
    node3 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a3, order=1, seq=mocker.MagicMock(spec=expression_transform)))

    a4 = ast.Num(n=3)
    node4 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a4, order=1, seq=mocker.MagicMock(spec=expression_transform)))

    # Next, the node that uses both of them.
    a5 = ast.Num(n=4)
    seq5 = mocker.MagicMock(spec=expression_transform)
    ast_name_dict: Dict[str, ast.AST] = {"a2": a2, "a3": a3, "a4": a4}
    seq5.render_ast.return_value = parse_ast_string("a2 + e1000", ast_name_dict)
    node5 = g.add_vertex(info=mock_vinfo(mocker, level=2, node=a5, order=1, seq=seq5))
    g.add_edge(node5, node2, info=e_info(True, 1))
    g.add_edge(node5, node3, info=e_info(False, 2))
    g.add_edge(node5, node4, info=e_info(False, 2))

    # Now do the iterator reduction. It should hit this node.
    reduce_iterator_chaining(g, level=2, qt=mock_qt)

    # However, the sequence should have changed into some thing that is a second derivation.
    new_seq5 = get_v_info(node5).sequence
    assert MatchAST("Select(a3, lambda e1000: a2 + e1000)", ast_name_dict) == new_seq5.render_ast({})
    seq5.render_ast.assert_called_with(MatchASTDict({a3: ast.Name(id='e1000')}))


def test_single_to_zero(mocker, mock_qt):
    'Knock a level 2 all the way down to level 0'
    g = Graph(directed=True)

    a1 = ast.Constant(10)
    seq_junk = mocker.MagicMock(spec=expression_transform)
    level_0 = g.add_vertex(info=mock_vinfo(mocker, node=a1, seq=seq_junk, order=0, level=0))

    a2 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=expression_transform)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, node=a2, seq=seq_met, order=0, level=2))

    g.add_edge(level_2, level_0, info=e_info(True, 1))

    run_linear_reduction(g, mock_qt)

    assert len(g.vs()) == 2
    assert get_v_info(level_2).level == 0


def test_double_nodes_reduced_to_zero(mocker, mock_qt):
    '2 nodes, dependent on same prior node, combined and reduced in right order'
    g = Graph(directed=True)
    a_1 = ast.Constant(1)
    level_1 = g.add_vertex(info=mock_vinfo(mocker, level=0, node=a_1, order=0, seq=mocker.MagicMock(spec=expression_transform)))

    a_2 = ast.Constant(1)
    level_2 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_2, order=1, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_2, level_1, info=e_info(True, 1))

    a_3 = ast.Constant(1)
    level_3 = g.add_vertex(info=mock_vinfo(mocker, level=1, node=a_3, order=2, seq=mocker.MagicMock(spec=expression_transform)))
    g.add_edge(level_3, level_1, info=e_info(True, 1))

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


def test_two_iterators_combined(mocker, mock_root_sequence_transform, mock_qt):
    'Make sure the common iterator code is called properly'

    g = Graph(directed=True)
    g['info'] = g_info([], iter_index=5)

    mine, a1, root_seq = mock_root_sequence_transform
    r = g.add_vertex(info=mock_vinfo(mocker, level=0, seq=root_seq, node=a1))

    n1 = g.add_vertex(info=mock_vinfo(mocker, level=1, order=1, seq=mocker.MagicMock(spec=expression_transform), node=ast.Name('a')))
    n2 = g.add_vertex(info=mock_vinfo(mocker, level=1, order=2, seq=mocker.MagicMock(spec=expression_transform), node=ast.Name('b')))
    g.add_edge(n1, r, info=e_info(True, 1))
    g.add_edge(n2, r, info=e_info(True, 2))

    a_adder = ast.Name('adder')
    n3 = g.add_vertex(info=mock_vinfo(mocker, level=1, order=3, seq=mocker.MagicMock(spec=expression_transform), node=a_adder))
    g.add_edge(n3, n1, info=e_info(True, 1))
    g.add_edge(n3, n2, info=e_info(False, 2))

    run_linear_reduction(g, mock_qt)

    # Dig out the expression that should have been combined.
    assert len(g.vs()) == 3
    v_list = [i_v for i_v in g.vs() if a_adder in get_v_info(i_v).node_as_dict]
    assert len(v_list) == 1
    v = v_list[0]
    seq = get_v_info(v).sequence

    # Two things going on - first, it should be at level 0, and so it should be wrapped once.
    assert get_v_info(v).level == 0
    assert isinstance(seq, sequence_downlevel)

    # Now - since we are coming from two different iterators, it should have another inside it - this is the chaining sequence.
    seq_chained = seq.transform
    assert isinstance(seq_chained, sequence_downlevel)
