import ast

from hep_tables.graph_info import e_info

import pytest
from hep_tables.util_ast import astIteratorPlaceholder
from hep_tables.util_graph import child_iterator_in_use, depth_first_traversal, get_iterator_index, parent_iterator_index
from igraph import Graph

from tests.conftest import mock_vinfo


def test_traversal_empty():
    g = Graph(directed=True)
    assert len(list(depth_first_traversal(g))) == 0


def test_traversal_one():
    g = Graph(directed=True)
    g.add_vertex()
    r = list(depth_first_traversal(g))
    assert len(r) == 1
    assert len(r[0]) == 1


def test_travesal_branch(mocker):
    g = Graph(directed=True)
    a1 = g.add_vertex(info=mock_vinfo(mocker, order=2))
    a2 = g.add_vertex(info=mock_vinfo(mocker, order=2))
    a3 = g.add_vertex(info=mock_vinfo(mocker, order=2))
    a4 = g.add_vertex(info=mock_vinfo(mocker, order=2))

    g.add_edges([(a2, a1), (a3, a1), (a4, a2), (a4, a3)])
    r = list(depth_first_traversal(g))
    assert len(r) == 3
    assert len(r[0]) == 1
    assert len(r[1]) == 2
    assert len(r[2]) == 1

    assert r[0][0] == a1
    assert r[2][0] == a4


def test_traversal_ordered_1(mocker):
    g = Graph(directed=True)
    a1 = g.add_vertex()
    a2 = g.add_vertex(info=mock_vinfo(mocker, order=1))
    a3 = g.add_vertex(info=mock_vinfo(mocker, order=2))

    g.add_edges([(a2, a1), (a3, a1)])
    r = list(depth_first_traversal(g))
    assert r[1][0] == a2
    assert r[1][1] == a3


def test_traversal_ordered_2(mocker):
    g = Graph(directed=True)
    a1 = g.add_vertex()
    a2 = g.add_vertex(info=mock_vinfo(mocker, order=2))
    a3 = g.add_vertex(info=mock_vinfo(mocker, order=1))

    g.add_edges([(a2, a1), (a3, a1)])
    r = list(depth_first_traversal(g))
    assert r[1][0] == a3
    assert r[1][1] == a2


def test_parent_iterator_single(mocker):
    g = Graph(directed=True)
    v = g.add_vertex(info=mock_vinfo(mocker, node={ast.Constant(10): astIteratorPlaceholder(1)}))

    assert parent_iterator_index(v) == 1


def test_parent_iterator_single_fail(mocker):
    g = Graph(directed=True)
    v = g.add_vertex(info=mock_vinfo(mocker, node={
        ast.Constant(10): astIteratorPlaceholder(1),
        ast.Constant(20): astIteratorPlaceholder(2),
    }))

    with pytest.raises(Exception):
        parent_iterator_index(v)


def test_parent_iterator_linked(mocker):
    g = Graph(directed=True)
    g.add_vertex(info=mock_vinfo(mocker, node={ast.Constant(10): astIteratorPlaceholder(1)}))
    v2 = g.add_vertex(info=mock_vinfo(mocker, node={ast.Constant(10): astIteratorPlaceholder(1)}))
    g.add_edge(v2, v2, info=e_info(True, 2))

    assert parent_iterator_index(v2) == 2


def test_child_itr_good_child(mocker):
    g = Graph(directed=True)
    v_p = g.add_vertex(info=mock_vinfo(mocker, level=1, node={ast.Constant(10): astIteratorPlaceholder(1)}))
    v = g.add_vertex(info=mock_vinfo(mocker, level=1, node={ast.Constant(10): astIteratorPlaceholder(2)}))
    g.add_edge(v, v_p, info=e_info(True, 1))

    assert child_iterator_in_use(v_p, 1) == 2


def test_child_itr_bad_level_child(mocker):
    g = Graph(directed=True)
    v_p = g.add_vertex(info=mock_vinfo(mocker, level=1, node={ast.Constant(10): astIteratorPlaceholder(1)}))
    v = g.add_vertex(info=mock_vinfo(mocker, level=2, node={ast.Constant(10): astIteratorPlaceholder(2)}))
    g.add_edge(v, v_p, info=e_info(True, 1))

    assert child_iterator_in_use(v_p, 1) is None


def test_child_itr_no_children(mocker):
    g = Graph(directed=True)
    v = g.add_vertex(info=mock_vinfo(mocker, node={ast.Constant(10): astIteratorPlaceholder(1)}))

    assert child_iterator_in_use(v, 1) is None


def test_get_iter_index_one(mocker):
    g = Graph(directed=True)
    v = g.add_vertex(info=mock_vinfo(mocker, node={ast.Constant(10): astIteratorPlaceholder(1)}))

    assert get_iterator_index(v) == 1


def test_get_iter_index_one_two(mocker):
    g = Graph(directed=True)
    v = g.add_vertex(info=mock_vinfo(mocker, node={ast.Constant(10): astIteratorPlaceholder(1), ast.Constant(10): astIteratorPlaceholder(2)}))

    with pytest.raises(Exception):
        get_iterator_index(v)


def test_get_iter_index_one_two(mocker):
    g = Graph(directed=True)
    v = g.add_vertex(info=mock_vinfo(mocker, node={ast.Constant(10): astIteratorPlaceholder(1), ast.Constant(10): astIteratorPlaceholder(1)}))

    assert get_iterator_index(v) == 1
