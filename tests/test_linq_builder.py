import ast
from tests.conftest import translate_linq
from anytree.node.anynode import AnyNode

from func_adl.event_dataset import EventDataset
from func_adl.object_stream import ObjectStream
from hep_tables.transforms import root_sequence_transform, sequence_transform
from hep_tables.linq_builder import build_linq_expression
from typing import Any
from dataframe_expressions import ast_DataFrame
from hep_tables import xaod_table


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
    top_level = AnyNode(node=a, seq=root_seq)

    r = build_linq_expression(top_level)
    assert r is mine


def test_source_and_single_generator(mocker):
    '''Return a sequence of met stuff'''
    mine, a1, root_seq = mock_root_sequence_transform(mocker)
    level_0 = AnyNode(node=a1, seq=root_seq)

    a2 = ast.Constant(10)
    seq_met = mocker.MagicMock(spec=sequence_transform)
    proper_return = mine.Select("lambda e1: e1.met")
    seq_met.sequence.return_value = proper_return
    level_1 = AnyNode(node=a2, seq=seq_met, parent=level_0)

    r = build_linq_expression(level_1)

    assert r is proper_return
    assert seq_met.sequence.called_with(mine)
