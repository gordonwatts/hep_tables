from typing import Iterable
from hep_tables.transforms import sequence_predicate_base
from hep_tables.graph_info import copy_v_info, get_v_info, v_info
import ast
from igraph import Graph


def test_node_info_basic(mocker):
    dummy_seq = mocker.MagicMock(spec=sequence_predicate_base)
    a = ast.Name(id='a')
    v_info(1, dummy_seq, Iterable[int], a)


def test_get_metadata(mocker):
    g = Graph(directed=True)
    v_i = mocker.MagicMock(spec=v_info)
    v = g.add_vertex(info=v_i)

    assert get_v_info(v) is v_i


def test_copy_v_info_no_param(mocker):
    v1 = v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], ast.Name(id='a'))
    v2 = copy_v_info(v1)
    assert v1 == v2


def test_copy_v_info_new_level(mocker):
    v1 = v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], ast.Name(id='a'))
    v2 = copy_v_info(v1, new_level=2)
    assert v1 != v2
    assert v1.level == 1
    assert v2.level == 2


def test_copy_v_info_new_seq(mocker):
    v1 = v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], ast.Name(id='a'))
    new_seq = mocker.MagicMock(spec=sequence_predicate_base)
    v2 = copy_v_info(v1, new_sequence=new_seq)
    assert v1 != v2
    assert v2.sequence == new_seq
