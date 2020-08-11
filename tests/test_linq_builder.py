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


def test_just_the_source(mocker, mock_qt):
    '''Simple single source node. This isn't actually valid LINQ that could be rendered.
    '''

    mine, a, root_seq = mock_root_sequence_transform(mocker)
    g = Graph(directed=True)
    g.add_vertex(node=a, seq=root_seq)

    r = build_linq_expression(g, mock_qt)
    root_seq.sequence.assert_called_with(None, {})
    assert r is mine


def compare_dict_ast_args(first: Dict[ast.AST, ast.AST],
                          second: Dict[ast.AST, ast.AST]) -> bool:
    if set(first.keys()) != set(second.keys()):
        print('Number of keys did not match')
        return False

    for k in first.keys():
        if ast.dump(first[k]) != ast.dump(second[k]):
            print(f'Key {ast.dump(k)} did not have matching first={ast.dump(first[k])} second={ast.dump(second[k])}')
            return False

    return True


class MatchSeqDict:
    def __init__(self, some_obj):
        self.some_obj = some_obj

    def __eq__(self, other):
        return compare_dict_ast_args(self.some_obj, other)


def test_source_and_single_generator(mocker, mock_qt):
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

    r = build_linq_expression(g, mock_qt)

    assert r is proper_return
    seq_met.sequence.assert_called_with(mine, MatchSeqDict({a1: astIteratorPlaceholder()}))


class MatchObjectSequence:
    def __init__(self, a_list: List[ast.AST]):
        from func_adl.ast.func_adl_ast_utils import change_extension_functions_to_calls
        self._asts = [change_extension_functions_to_calls(a) for a in a_list]

    def clean(self, a: ast.AST):
        return ast.dump(a) \
            .replace(', annotation=None', '') \
            .replace(', vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]', '') \
            .replace(', ctx=Load()', '')

    def __eq__(self, other: ObjectStream):
        other_ast = self.clean(other._ast)
        r = any(self.clean(a) == other_ast for a in self._asts)
        if not r:
            print(f'test: {self.clean(other._ast)}')
            for a in self._asts:
                print(f'true: {self.clean(a)}')
        return r


def test_two_source_operator(mocker, mock_qt):
    '''An operation that splits into two and then combines back into one'''

    mine, a1, root_seq = mock_root_sequence_transform(mocker)
    g = Graph(directed=True)
    level_0 = g.add_vertex(node=a1, seq=root_seq)

    a2_1 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=sequence_transform)
    proper_return2_1 = ObjectStream(ast.Name(id='a')).Select("lambda e1: e1.met")
    seq_met.sequence.return_value = proper_return2_1
    level_1_1 = g.add_vertex(node=a2_1, seq=seq_met)
    g.add_edge(level_1_1, level_0, main_seq=True)

    a2_2 = ast.Constant(20)
    seq_met_prime = mocker.MagicMock(spec=sequence_transform)
    proper_return2_2 = ObjectStream(ast.Name(id='b')).Select("lambda e1: e1.met_prime")
    seq_met_prime.sequence.return_value = proper_return2_2
    level_1_2 = g.add_vertex(node=a2_2, seq=seq_met_prime)
    g.add_edge(level_1_2, level_0, main_seq=True)

    a3 = ast.Constant(30)
    seq_combine = mocker.MagicMock(spec=sequence_transform)
    proper_return3 = ObjectStream(ast.Name(id='c')).Select("lambda e: e[0] + e[1]")
    seq_combine.sequence.return_value = proper_return3
    level_2 = g.add_vertex(node=a3, seq=seq_combine)
    g.add_edge(level_2, level_1_1, main_seq=True)
    g.add_edge(level_2, level_1_2, main_seq=False)

    r = build_linq_expression(g, mock_qt)
    assert r is not None

    root_seq.sequence.assert_called_with(None, {})

    seq_met.sequence.assert_called_with(MatchObjectSequence([ast.Name(id='e1000')]), MatchSeqDict({a1: ast.Name(id='e1000')}))
    seq_met_prime.sequence.assert_called_with(MatchObjectSequence([ast.Name(id='e1000')]), MatchSeqDict({a1: ast.Name(id='e1000')}))

    two_returns_1 = mine \
        .Select("lambda e1000: (a.Select(lambda e1: e1.met), b.Select(lambda e1: e1.met_prime))")
    two_returns_2 = mine \
        .Select("lambda e1000: (b.Select(lambda e1: e1.met_prime), a.Select(lambda e1: e1.met))")

    seq_combine.sequence.assert_called_with(
        MatchObjectSequence([two_returns_1._ast, two_returns_2._ast]),
        MatchSeqDict(
            {
                a2_1: ast.Subscript(value=astIteratorPlaceholder(), slice=ast.Index(0)),
                a2_2: ast.Subscript(value=astIteratorPlaceholder(), slice=ast.Index(1)),
            })
    )

    assert r is proper_return3


def test_two_source_twice_operator(mocker, mock_qt):
    'An operation that splits in two, and stays split in two for another generation.'

    mine, a1, root_seq = mock_root_sequence_transform(mocker)
    g = Graph(directed=True)
    level_0 = g.add_vertex(node=a1, seq=root_seq)

    a2_1 = ast.Constant(10)
    seq_met21 = mocker.MagicMock(spec=sequence_transform)
    proper_return2_1 = ObjectStream(ast.Name(id='a')).Select("lambda e1: e1.met21")
    seq_met21.sequence.return_value = proper_return2_1
    level_2_1 = g.add_vertex(node=a2_1, seq=seq_met21)
    g.add_edge(level_2_1, level_0, main_seq=True)

    a2_2 = ast.Constant(20)
    seq_met22 = mocker.MagicMock(spec=sequence_transform)
    proper_return2_2 = ObjectStream(ast.Name(id='b')).Select("lambda e1: e1.met22")
    seq_met22.sequence.return_value = proper_return2_2
    level_2_2 = g.add_vertex(node=a2_2, seq=seq_met22)
    g.add_edge(level_2_2, level_0, main_seq=True)

    a3_1 = ast.Constant(10)
    seq_met31 = mocker.MagicMock(spec=sequence_transform)
    proper_return3_1 = ObjectStream(ast.Name(id='c')).Select("lambda e1: e1.met31")
    seq_met31.sequence.return_value = proper_return3_1
    level_3_1 = g.add_vertex(node=a3_1, seq=seq_met31)
    g.add_edge(level_3_1, level_2_1, main_seq=True)

    a3_2 = ast.Constant(20)
    seq_met32 = mocker.MagicMock(spec=sequence_transform)
    proper_return2_2 = ObjectStream(ast.Name(id='d')).Select("lambda e1: e1.met32")
    seq_met32.sequence.return_value = proper_return2_2
    level_3_2 = g.add_vertex(node=a3_2, seq=seq_met32)
    g.add_edge(level_3_2, level_2_2, main_seq=True)

    a4 = ast.Constant(30)
    seq_combine = mocker.MagicMock(spec=sequence_transform)
    proper_return4 = ObjectStream(ast.Name(id='e')).Select("lambda e: e[0] + e[1]")
    seq_combine.sequence.return_value = proper_return4
    level_4 = g.add_vertex(node=a4, seq=seq_combine)
    g.add_edge(level_4, level_3_1, main_seq=True)
    g.add_edge(level_4, level_3_2, main_seq=False)

    r = build_linq_expression(g, mock_qt)
    assert r is not None

    two_returns_1 = mine \
        .Select("lambda e1000: (a.Select(lambda e1: e1.met21), b.Select(lambda e1: e1.met22))") \
        .Select("lambda e1000: (d.Select(lambda e1: e1.met32), c.Select(lambda e1: e1.met31))")
    two_returns_2 = mine \
        .Select("lambda e1000: (b.Select(lambda e1: e1.met22), a.Select(lambda e1: e1.met21))") \
        .Select("lambda e1000: (c.Select(lambda e1: e1.met31), d.Select(lambda e1: e1.met32))")

    seq_combine.sequence.assert_called_with(
        MatchObjectSequence([two_returns_1._ast, two_returns_2._ast]),
        MatchSeqDict(
            {
                a3_1: ast.Subscript(value=astIteratorPlaceholder(), slice=ast.Index(1)),
                a3_2: ast.Subscript(value=astIteratorPlaceholder(), slice=ast.Index(0)),
            })
    )

    assert r is proper_return4
