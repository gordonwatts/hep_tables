import ast

from dataframe_expressions import ast_DataFrame
from dataframe_expressions.render import render
from func_adl import EventDataset

from hep_tables import xaod_table
from hep_tables.utils import _find_dataframes, _find_root_expr

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
