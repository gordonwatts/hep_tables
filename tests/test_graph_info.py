import ast
from typing import Iterable

import pytest
from igraph import Graph

from hep_tables.graph_info import copy_v_info, get_v_info, v_info
from hep_tables.transforms import sequence_predicate_base
from hep_tables.util_ast import astIteratorPlaceholder


def test_node_info_basic(mocker):
    dummy_seq = mocker.MagicMock(spec=sequence_predicate_base)
    a = ast.Name(id='a')
    v_info(1, dummy_seq, Iterable[int], {a: astIteratorPlaceholder(1)})


def test_get_metadata(mocker):
    g = Graph(directed=True)
    v_i = mocker.MagicMock(spec=v_info)
    v = g.add_vertex(info=v_i)

    assert get_v_info(v) is v_i


def test_copy_v_info_no_param(mocker):
    v1 = v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {ast.Name(id='a'): astIteratorPlaceholder(1)})
    v2 = copy_v_info(v1)
    assert v1 == v2


def test_copy_v_info_new_level(mocker):
    v1 = v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {ast.Name(id='a'): astIteratorPlaceholder(1)})
    v2 = copy_v_info(v1, new_level=2)
    assert v1 != v2
    assert v1.level == 1
    assert v2.level == 2


def test_copy_v_info_new_seq(mocker):
    v1 = v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {ast.Name(id='a'): astIteratorPlaceholder(1)})
    new_seq = mocker.MagicMock(spec=sequence_predicate_base)
    v2 = copy_v_info(v1, new_sequence=new_seq)
    assert v1 != v2
    assert v2.sequence == new_seq


def test_get_node_ast_dict_with_single(mocker):
    a = ast.Name(id='a')
    v1 = v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: astIteratorPlaceholder(1)})
    d = v1.node_as_dict
    assert len(d) == 1
    assert a in d
    assert isinstance(d[a], astIteratorPlaceholder)


def test_info_with_dict_single(mocker):
    a = ast.Name(id='a')
    a_resolved = ast.Name(id='b')
    v1 = v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float], {a: a_resolved})
    assert v1.node is a
    assert v1.node_as_dict[a] is a_resolved


def test_info_with_multiple_is_bad(mocker):
    v1 = v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float],
                {
                    ast.Constant(10): ast.Constant(10),
                    ast.Constant(20): ast.Name(id='hi')
    })
    with pytest.raises(Exception):
        v1.node


def test_info_with_multiple(mocker):
    v1 = v_info(1, mocker.MagicMock(spec=sequence_predicate_base), Iterable[float],
                {
                    ast.Constant(10): ast.Constant(10),
                    ast.Constant(20): ast.Name(id='hi')
    })
    assert len(v1.node_as_dict) == 2
