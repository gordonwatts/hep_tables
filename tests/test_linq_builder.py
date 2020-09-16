import ast
from tests.conftest import MatchASTDict
from typing import Any, Iterable

from igraph import Graph

from hep_tables.graph_info import e_info, g_info, v_info
from hep_tables.linq_builder import build_linq_expression
from hep_tables.transforms import sequence_predicate_base
from hep_tables.util_ast import astIteratorPlaceholder


def test_just_the_source(mock_root_sequence_transform):
    '''Simple single source node. This isn't actually valid LINQ that could be rendered.
    '''

    mine, a, root_seq = mock_root_sequence_transform
    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, root_seq, Any, a))

    r = build_linq_expression(g)
    root_seq.sequence.assert_called_with(None, {})
    assert r is mine


def test_source_and_single_generator(mocker, mock_root_sequence_transform):
    '''Two sequence statements linked together'''
    mine, a1, root_seq = mock_root_sequence_transform
    g = Graph(directed=True)
    level_0 = g.add_vertex(info=v_info(1, root_seq, Any, a1))

    a2 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=sequence_predicate_base)
    proper_return = mine.Select("lambda e1: e1.met")
    seq_met.sequence.return_value = proper_return
    level_1 = g.add_vertex(info=v_info(1, seq_met, Any, a2))

    g.add_edge(level_1, level_0, info=e_info(True, 1))

    r = build_linq_expression(g)

    assert r is proper_return
    seq_met.sequence.assert_called_with(mine, MatchASTDict({a1: astIteratorPlaceholder([None])}))


def test_level_appropriate(mocker, mock_root_sequence_transform):
    'Make sure we call with an extra level down in the ast placeholder'
    mine, a1, root_seq = mock_root_sequence_transform
    g = Graph(directed=True)
    level_0 = g.add_vertex(info=v_info(1, root_seq, Any, {a1: astIteratorPlaceholder([1])}))

    a2 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=sequence_predicate_base)
    proper_return = mine.Select("lambda e1: e1.met")
    seq_met.sequence.return_value = proper_return
    level_1 = g.add_vertex(info=v_info(1, seq_met, Any, a2))

    g.add_edge(level_1, level_0, info=e_info(True, 1))

    build_linq_expression(g)

    passed_dict = seq_met.sequence.call_args[0][1]
    assert len(passed_dict[a1].levels) == 2
