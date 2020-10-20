from hep_tables.constant import Constant


def test_instant():
    Constant[int]  # type: ignore


def test_identity():
    assert Constant[int] is Constant[int]  # type: ignore
    assert Constant[int] == Constant[int]  # type: ignore
    assert Constant[int].type is int  # type: ignore


def test_is_constant():
    assert Constant.isconstant(Constant[int])  # type: ignore
    assert not Constant.isconstant(int)


def test_const_type():
    assert Constant.constanttype(Constant[int]) == int  # type: ignore
