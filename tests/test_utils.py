import ast

from dataframe_expressions import ast_DataFrame
from dataframe_expressions.render import render
from func_adl import EventDataset
import pytest

from hep_tables import xaod_table
from hep_tables.utils import (
    _find_dataframes, _find_root_expr, _index_text_tuple, _parse_elements)

# For use in testing - a mock.
f = EventDataset('locads://bogus')


def test_find_dataframes():
    df = xaod_table(f)
    seq = df.jets.pt
    expr, _ = render(seq)

    found_df = _find_dataframes(expr)
    assert isinstance(found_df, ast_DataFrame)
    assert found_df.dataframe is df


def test_find_nested_dataframes():
    df = xaod_table(f)
    seq = df.jets[df.jets.pt > 30].pt
    expr, _ = render(seq)

    found_df = _find_dataframes(expr)
    assert isinstance(found_df, ast_DataFrame)
    assert found_df.dataframe is df


def test_find_root_ast_df_nested():
    df = xaod_table(f)
    a = ast_DataFrame(df)

    attr = ast.Attribute(value=a, attr='jets', ctx=ast.Load())

    r = _find_root_expr(attr, ast.Num(n=10, ctx=ast.Load()))
    assert r is not None
    assert r is a


def test_find_root_ast_df_simple():
    df = xaod_table(f)
    a = ast_DataFrame(df)

    r = _find_root_expr(a, ast.Num(n=10, ctx=ast.Load()))
    assert r is not None
    assert r is a


def test_find_root_ast():
    df = xaod_table(f)
    a = ast_DataFrame(df)

    attr = ast.Attribute(value=a, attr='jets', ctx=ast.Load())
    attr1 = ast.Attribute(value=attr, attr='dude', ctx=ast.Load())

    r = _find_root_expr(attr1, attr)
    assert r is not None
    assert r is attr


def test_find_root_in_function():
    df = xaod_table(f)
    a = ast_DataFrame(df)

    attr = ast.Attribute(value=a, attr='jets', ctx=ast.Load())
    call = ast.Call(func=ast.Name(id='sin'), args=[attr], keywords=None)

    r = _find_root_expr(call, attr)
    assert r is attr


def test_find_root_arg_not_right():
    df = xaod_table(f)
    a = ast_DataFrame(df)

    # df.ele.deltar(df.jets), with df.jets as the arg.

    jets_attr = ast.Attribute(value=a, attr='jets', ctx=ast.Load())
    eles_attr = ast.Attribute(value=a, attr='eles', ctx=ast.Load())

    call = ast.Call(func=eles_attr, args=[jets_attr], keywords=None)

    r = _find_root_expr(call, jets_attr)
    assert r is a


def test_no_splits():
    s = _parse_elements('hi')
    assert len(s) == 1
    assert s[0] == 'hi'


def test_simple_split():
    s = _parse_elements('(hi,there)')
    assert len(s) == 2
    assert s[0] == 'hi'
    assert s[1] == 'there'


def test_split_not_parans():
    s = _parse_elements('hi, there')
    assert len(s) == 1


def test_split_with_func():
    s = _parse_elements('(sin(20), e1)')
    assert len(s) == 2
    assert s[0] == 'sin(20)'
    assert s[1] == ' e1'


def test_split_with_2arg_func():
    s = _parse_elements('(asing(x,y),e2)')
    assert len(s) == 2
    assert s[0] == 'asing(x,y)'
    assert s[1] == 'e2'


def test_text_tuple_none():
    assert _index_text_tuple('e5', 1) == 'e5[1]'


def test_text_tuple_good():
    assert _index_text_tuple('(e5,e6)', 0) == 'e5'


def test_text_tuple_bad():
    with pytest.raises(Exception):
        _index_text_tuple('(e5,e6)', 2)

# def test_fail_to_find_two_dataframes():
#     df1 = xaod_table(f)
#     f2 = EventDataset('locads://bogusss')
#     df2 = xaod_table(f2)

#     deq = df1.jets[df2.jets.pt > 30].pt
#     expr = render(deq)

#     with pytest.raises(Exception):
#         _find_dataframes(expr)
# TODO: this doesn't fail yet, but it should - error seems to be in the underlying library.
# Since this is a prototype, no need to chase it down right now.
