import ast
from hep_tables.graph_info import e_info, v_info
from typing import Any, Dict

from igraph import Graph

from hep_tables.linq_builder import build_linq_expression
from hep_tables.transforms import astIteratorPlaceholder, sequence_transform


class MatchASTDict:
    def __init__(self, true_dict: Dict[ast.AST, ast.AST]):
        self._true = true_dict

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, dict):
            return False

        if len(o) != len(self._true):
            return False

        if set(o.keys()) != set(self._true.keys()):
            return False

        return all(ast.dump(o[k]) == ast.dump(self._true[k]) for k in self._true.keys())


def test_just_the_source(mock_root_sequence_transform):
    '''Simple single source node. This isn't actually valid LINQ that could be rendered.
    '''

    mine, a, root_seq = mock_root_sequence_transform
    g = Graph(directed=True)
    g.add_vertex(info=v_info(1, root_seq, Any, a))

    r = build_linq_expression(g)
    root_seq.sequence.assert_called_with(None, MatchASTDict({a: astIteratorPlaceholder()}))
    assert r is mine


def test_source_and_single_generator(mocker, mock_root_sequence_transform):
    '''Two sequence statements linked together'''
    mine, a1, root_seq = mock_root_sequence_transform
    g = Graph(directed=True)
    level_0 = g.add_vertex(info=v_info(1, root_seq, Any, a1))

    a2 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=sequence_transform)
    proper_return = mine.Select("lambda e1: e1.met")
    seq_met.sequence.return_value = proper_return
    level_1 = g.add_vertex(info=v_info(1, seq_met, Any, a2))

    g.add_edge(level_1, level_0, info=e_info(True))

    r = build_linq_expression(g)

    assert r is proper_return
    seq_met.sequence.assert_called_with(mine, MatchASTDict({a2: astIteratorPlaceholder()}))
