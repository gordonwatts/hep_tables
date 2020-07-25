import ast
from typing import Dict, List

import pytest
from dataframe_expressions import ast_DataFrame, render
from func_adl_xAOD import ServiceXDatasetSource

from hep_tables import xaod_table
from hep_tables.utils import (QueryVarTracker, _count_list, _find_dataframes,
                              _find_root_expr, _index_text_tuple, _is_list,
                              _is_of_type, _parse_elements, _type_replace,
                              _unwrap_list, find_common_sub_expressions)


def test_find_dataframes(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt
    expr, _ = render(seq)

    found_df = _find_dataframes(expr)
    assert isinstance(found_df, ast_DataFrame)
    assert found_df.dataframe is df


def test_find_nested_dataframes(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets[df.jets.pt > 30].pt
    expr, _ = render(seq)

    found_df = _find_dataframes(expr)
    assert isinstance(found_df, ast_DataFrame)
    assert found_df.dataframe is df


def test_find_root_ast_df_nested(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    a = ast_DataFrame(df)

    attr = ast.Attribute(value=a, attr='jets', ctx=ast.Load())

    r = _find_root_expr(attr, ast.Num(n=10, ctx=ast.Load()))
    assert r is not None
    assert r is a


def test_find_root_ast_df_simple(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    a = ast_DataFrame(df)

    r = _find_root_expr(a, ast.Num(n=10, ctx=ast.Load()))
    assert r is not None
    assert r is a


def test_find_root_no_root(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    a = ast_DataFrame(df)

    r = _find_root_expr(ast.Num(n=10, ctx=ast.Load()), a)
    assert r is None


def test_find_root_ast(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    a = ast_DataFrame(df)

    attr = ast.Attribute(value=a, attr='jets', ctx=ast.Load())
    attr1 = ast.Attribute(value=attr, attr='dude', ctx=ast.Load())

    r = _find_root_expr(attr1, attr)
    assert r is not None
    assert r is attr


def test_find_root_in_function(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    a = ast_DataFrame(df)

    attr = ast.Attribute(value=a, attr='jets', ctx=ast.Load())
    call = ast.Call(func=ast.Name(id='sin'), args=[attr], keywords=None)

    r = _find_root_expr(call, attr)
    assert r is attr


def test_find_root_arg_not_right(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
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


def test_is_list_list():
    assert _is_list(List[object])


def test_is_list_not_list():
    assert not _is_list(object)


def test_is_list_double_list():
    assert _is_list(List[List[object]])


def test_is_list_not_bool():
    assert not _is_list(bool)


def test_unwrap_list():
    assert _unwrap_list(List[object]) is object


def test_unwrap_list_of_list():
    assert _unwrap_list(List[List[int]]) is List[int]


def test_type_replace_nogo():
    assert _type_replace(int, List[int], int) is None


def test_type_replace_terminal():
    assert _type_replace(int, int, float) is float


def test_type_replace_list():
    assert _type_replace(List[int], List[int], int) is int


def test_type_replace_list_object():
    assert _type_replace(List[int], List[object], int) is int


def test_type_replace_nested_list():
    assert _type_replace(List[List[int]], List[object], int) is List[int]


def test_is_type_not():
    assert not _is_of_type(int, List[int])


def test_is_type_object():
    assert _is_of_type(int, object)


def test_is_type_list():
    assert _is_of_type(List[int], List[int])


def test_is_type_list_obj():
    assert _is_of_type(List[int], List[object])


def test_is_type_list_dict_obj():
    assert not _is_of_type(Dict[int, str], List[object])


def test_is_type_list_diff():
    assert not _is_of_type(List[int], List[float])


def test_is_type_list_not_object():
    assert not _is_of_type(List[int], object)


def test_is_type_list_not_object_inverse():
    assert not _is_of_type(object, List[int])


def test_count_list_none():
    assert _count_list(int) == 0


def test_count_list_one():
    assert _count_list(List[int]) == 1


def test_count_list_two():
    assert _count_list(List[List[int]]) == 2


def test_new_var_unique():
    q = QueryVarTracker()
    v1 = q.new_var_name()
    v2 = q.new_var_name()
    assert v1 != v2


def test_new_var_type():
    q = QueryVarTracker()
    v1 = q.new_term(type(int))

    assert v1.type == type(int)


def test_common_no_asts():
    assert find_common_sub_expressions([]) is None


def test_common_literal():
    a = ast.parse('1+1').body[0].value  # type: ignore
    assert find_common_sub_expressions([a]) is None


def test_common_single_xaod(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    expr, _ = render(df.jets.pt)
    assert find_common_sub_expressions([expr]) is expr


def test_common_single_xaod_and_literal(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    expr1, _ = render(df.jets.pt)

    expr2 = ast.parse('1+1').body[0].value  # type: ignore

    assert find_common_sub_expressions([expr1, expr2]) is expr1


def test_common_single_xaod_and_common(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    expr1, render_context = render(df.jets.pt)

    expr2, _ = render(df.jets.eta, render_context)

    common_expr = find_common_sub_expressions([expr1, expr2])

    assert isinstance(expr1, ast.Attribute)
    assert common_expr is expr1.value

    assert isinstance(expr2, ast.Attribute)
    assert common_expr is expr2.value


def test_common_single_xaod_and_identical(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    expr1, render_context = render(df.jets.pt)

    expr2, _ = render(df.jets.pt, render_context)

    common_expr = find_common_sub_expressions([expr1, expr2])

    assert common_expr is expr1


def test_common_single_xaod_depth_and_common(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    expr1, render_context = render(df.jets.pt)

    expr2, _ = render(df.jets.eta / 2.0, render_context)

    common_expr = find_common_sub_expressions([expr1, expr2])

    assert isinstance(expr1, ast.Attribute)
    assert common_expr is expr1.value


def test_common_single_xaod_rdepth_and_common(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    expr1, render_context = render(df.jets.pt)

    expr2, _ = render(df.jets.eta / 2.0, render_context)

    common_expr = find_common_sub_expressions([expr2, expr1])

    assert isinstance(expr1, ast.Attribute)
    assert common_expr is expr1.value


def test_common_single_2xaod(servicex_ds):
    f1 = ServiceXDatasetSource(servicex_ds)
    df1 = xaod_table(f1)
    expr1, _ = render(df1.jets.pt)

    f2 = ServiceXDatasetSource(servicex_ds)
    df2 = xaod_table(f2)
    expr2, _ = render(df2.jets.pt)
    assert find_common_sub_expressions([expr1, expr2]) is None
