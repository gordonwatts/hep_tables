import ast
from tests.conftest import MatchAST
from typing import Dict
from hep_tables.util_ast import add_level_to_holder, astIteratorPlaceholder, reduce_holder_by_level, replace_holder, set_holder_level_index
import pytest


def test_ast_ctor():
    a = astIteratorPlaceholder(1)
    assert len(a.levels) == 0


def test_add_one_level():
    a = astIteratorPlaceholder(1)
    a.set_level_index(2)
    assert len(a.levels) == 0
    assert a.new_level == 2
    new_a = a.next_level()

    assert len(new_a.levels) == 1
    assert new_a.levels[0] == 2
    assert len(a.levels) == 0


def test_level_no_index():
    a = astIteratorPlaceholder(1)
    new_a = a.next_level()
    assert len(new_a.levels) == 1
    assert new_a.levels[0] is None


def test_level_twice_fail():
    a = astIteratorPlaceholder(1)
    a.set_level_index(2)
    with pytest.raises(Exception):
        a.set_level_index(3)


def test_ast_dump():
    a = astIteratorPlaceholder(1)
    assert ast.dump(a) == 'astIteratorPlaceholder(itr_idx=1, level_index=[])'


def test_ast_dump_with_deep():
    a = astIteratorPlaceholder(4, [1, 2])
    assert ast.dump(a) == 'astIteratorPlaceholder(itr_idx=4, level_index=[1, 2])'


def test_set_holder_level_index():
    a = astIteratorPlaceholder(1)
    t = ast.Tuple(elts=[a])

    set_holder_level_index(1, 2).visit(t)
    new_t = add_level_to_holder().visit(t)

    new_a = new_t.elts[0]
    assert isinstance(new_a, astIteratorPlaceholder)
    assert len(new_a.levels) == 1
    assert new_a.levels[0] == 2


# def test_set_holder_level_index_wrong_id():
#     a = astIteratorPlaceholder(1)
#     t = ast.Tuple(elts=[a])

#     set_holder_level_index(2, 2).visit(t)
#     new_t = add_level_to_holder(2).visit(t)

#     new_a = new_t.elts[0]
#     assert isinstance(new_a, astIteratorPlaceholder)
#     assert len(new_a.levels) == 0


def test_ast_holder_reduce_easy():
    a = ast.Num(n=10)
    d: Dict[ast.AST, ast.AST] = {a: astIteratorPlaceholder(1, [0, 1])}

    new_d = reduce_holder_by_level(1, d)

    assert a in new_d
    a_ref = new_d[a]
    assert isinstance(a_ref, astIteratorPlaceholder)
    assert a_ref.levels == [0]


def test_ast_holder_reduce_wrong_id():
    a = ast.Num(n=10)
    d: Dict[ast.AST, ast.AST] = {a: astIteratorPlaceholder(1, [0, 1])}

    new_d = reduce_holder_by_level(2, d)

    assert a in new_d
    a_ref = new_d[a]
    assert isinstance(a_ref, astIteratorPlaceholder)
    assert a_ref.levels == [0, 1]


def test_ast_holder_reduce_empty():
    a = ast.Num(n=10)
    d: Dict[ast.AST, ast.AST] = {a: astIteratorPlaceholder(1)}

    new_d = reduce_holder_by_level(1, d)

    a_ref = new_d[a]
    assert isinstance(a_ref, astIteratorPlaceholder)
    assert a_ref.levels == []


def test_ast_holder_reduce_unchanged():
    a = ast.Num(n=10)
    r = astIteratorPlaceholder(1, [0, 1])
    d: Dict[ast.AST, ast.AST] = {a: r}

    new_d = reduce_holder_by_level(1, d)

    assert new_d[a] is not r


def test_ast_holder_reduce_burried():
    a = ast.Num(n=10)
    d: Dict[ast.AST, ast.AST] = {a: ast.Attribute(value=astIteratorPlaceholder(1, [0, 1]), attr='hi')}

    new_d = reduce_holder_by_level(1, d)
    attr = new_d[a]
    assert isinstance(attr, ast.Attribute)
    a_ref: astIteratorPlaceholder = attr.value  # type: ignore
    assert isinstance(a_ref, astIteratorPlaceholder)
    assert a_ref.levels == [0]


def test_replace_holder_with_var_no_levels():
    a = astIteratorPlaceholder(1)
    new_a = replace_holder(1, 'dude').visit(a)
    assert MatchAST("dude") == new_a


def test_replace_holder_with_var_wrong_itr_idx():
    a = astIteratorPlaceholder(1)
    new_a = replace_holder(2, 'dude').visit(a)
    assert new_a is a


def test_replace_holder_with_levels():
    a = astIteratorPlaceholder(1, [0, 1])
    new_a = replace_holder(1, 'dude').visit(a)
    assert MatchAST("dude[1]") == new_a


def test_replace_holder_with_none():
    a = astIteratorPlaceholder(1, [0, None])
    new_a = replace_holder(1, 'dude').visit(a)
    assert MatchAST("dude") == new_a


def test_replace_burried_holder():
    a = ast.Attribute(value=astIteratorPlaceholder(1, [0]), attr='fork')
    new_a = replace_holder(1, 'dude').visit(a)
    assert MatchAST("dude[0].fork") == new_a
