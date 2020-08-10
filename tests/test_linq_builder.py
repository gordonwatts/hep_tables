import ast

from func_adl.event_dataset import EventDataset
from func_adl.object_stream import ObjectStream
from hep_tables.transforms import astIteratorPlaceholder, root_sequence_transform, sequence_predicate_base, sequence_transform
from hep_tables.linq_builder import _monad_select_transform, build_linq_expression, depth_first_traversal
from typing import Any, Dict, List, Match
from dataframe_expressions import ast_DataFrame
from hep_tables import xaod_table
from igraph import Graph


def test_traversal_empty():
    g = Graph(directed=True)
    assert len(list(depth_first_traversal(g))) == 0


def test_traversal_one():
    g = Graph(directed=True)
    g.add_vertex()
    r = list(depth_first_traversal(g))
    assert len(r) == 1
    assert len(r[0]) == 1


def test_travesal_branch():
    g = Graph(directed=True)
    a1 = g.add_vertex()
    a2 = g.add_vertex()
    a3 = g.add_vertex()
    a4 = g.add_vertex()

    g.add_edges([(a2, a1), (a3, a1), (a4, a2), (a4, a3)])
    r = list(depth_first_traversal(g))
    assert len(r) == 3
    assert len(r[0]) == 1
    assert len(r[1]) == 2
    assert len(r[2]) == 1

    assert r[0][0] == a1
    assert r[2][0] == a4


class my_events(EventDataset):
    '''Dummy event source'''
    async def execute_result_async(self, a: ast.AST) -> Any:
        pass


def mock_root_sequence_transform(mocker):
    mine = my_events()
    a = ast_DataFrame(xaod_table(mine))

    root_seq = mocker.MagicMock(spec=root_sequence_transform)
    root_seq.sequence.return_value = mine

    return mine, a, root_seq


def test_just_the_source(mocker):
    '''Simple single source node. This isn't actually valid LINQ that could be rendered.
    '''

    mine, a, root_seq = mock_root_sequence_transform(mocker)
    g = Graph(directed=True)
    g.add_vertex(node=a, seq=root_seq)

    r = build_linq_expression(g)
    root_seq.sequence.assert_called_with(None, {})
    assert r is mine


def compare_dict_ast_args(first: Dict[ast.AST, ast.AST],
                          second: Dict[ast.AST, ast.AST]) -> bool:
    if set(first.keys()) != set(second.keys()):
        return False

    for k in first.keys():
        if ast.dump(first[k]) != ast.dump(second[k]):
            return False

    return True


class MatchSeqDict:
    def __init__(self, some_obj):
        self.some_obj = some_obj

    def __eq__(self, other):
        return compare_dict_ast_args(self.some_obj, other)


def test_source_and_single_generator(mocker):
    '''Return a sequence of met stuff'''
    mine, a1, root_seq = mock_root_sequence_transform(mocker)
    g = Graph(directed=True)
    level_0 = g.add_vertex(node=a1, seq=root_seq)

    a2 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=sequence_transform)
    proper_return = mine.Select("lambda e1: e1.met")
    seq_met.sequence.return_value = proper_return
    level_1 = g.add_vertex(node=a2, seq=seq_met)

    g.add_edge(level_1, level_0)

    r = build_linq_expression(g)

    assert r is proper_return
    seq_met.sequence.assert_called_once()
    seq_met.sequence.assert_called_with(mine, MatchSeqDict({a1: astIteratorPlaceholder()}))


class MatchMonandTransform:
    def __init__(self, statements: List[ObjectStream]):
        self._statements = statements

    def __eq__(self, other):
        assert isinstance(other, _monad_select_transform)
        assert len(self._statements) == len(other._tuple_statements)
        for s1, s2 in zip(self._statements, other._tuple_statements):
            if s1 is not s2:
                return False
        return True


def test_two_source_operator(mocker):
    '''Return a sequence that creates a tuple of two values'''

    mine, a1, root_seq = mock_root_sequence_transform(mocker)
    g = Graph(directed=True)
    level_0 = g.add_vertex(node=a1, seq=root_seq)

    a2_1 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=sequence_transform)
    proper_return2_1 = mine.Select("lambda e1: e1.met")
    seq_met.sequence.return_value = proper_return2_1
    level_1_1 = g.add_vertex(node=a2_1, seq=seq_met)
    g.add_edge(level_1_1, level_0)

    a2_2 = ast.Constant(20)
    seq_met_prime = mocker.MagicMock(spec=sequence_transform)
    proper_return2_2 = mine.Select("lambda e1: e1.met_prime")
    seq_met_prime.sequence.return_value = proper_return2_2
    level_1_2 = g.add_vertex(node=a2_2, seq=seq_met_prime)
    g.add_edge(level_1_2, level_0)

    a3 = ast.Constant(30)
    seq_combine = mocker.MagicMock(spec=sequence_transform)
    proper_return3 = mine.Select("lambda e: e[0] + e[1]")
    seq_combine.sequence.return_value = proper_return3
    level_2 = g.add_vertex(node=a3, seq=seq_combine)
    g.add_edge(level_2, level_1_1)
    g.add_edge(level_2, level_1_2)

    r = build_linq_expression(g)
    assert r is not None

    root_seq.sequence.assert_called_with(None, {})

    seq_met.sequence.assert_called_with(mine, MatchSeqDict({a1: astIteratorPlaceholder()}))
    seq_met_prime.sequence.assert_called_with(mine, MatchSeqDict({a1: astIteratorPlaceholder()}))

    seq_combine.sequence.assert_called_with(
        MatchMonandTransform([proper_return2_1, proper_return2_2]),
        MatchSeqDict(
        {
            a2_1: proper_return2_1,
            a2_2: proper_return2_2
        })
    )

    assert r is proper_return3
