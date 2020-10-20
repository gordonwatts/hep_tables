from hep_tables.constant import Constant, ConstantMeta


def test_instant():
    Constant[int]


def test_identity():
    assert Constant[int] is Constant[int]
    assert Constant[int] == Constant[int]
    assert Constant[int].type is int


def test_is_constant():
    assert Constant.isconstant(Constant[int])
    assert not Constant.isconstant(int)


def test_const_type():
    assert Constant.constanttype(Constant[int]) == int
