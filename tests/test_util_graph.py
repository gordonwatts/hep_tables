from tests.conftest import mock_vinfo
from igraph import Graph
from hep_tables.util_graph import depth_first_traversal


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
