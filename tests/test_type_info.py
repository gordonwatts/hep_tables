from typing import Callable, Iterable, Union

import pytest
from hep_tables.type_info import type_inspector


def test_attr_simple():

    class my_class:
        def pt(self) -> float:
            ...

    assert type_inspector().attribute_type(my_class, "pt") == Callable[[], float]


def test_attr_one_arg():
    class my_class:
        def pt(self, y: float) -> float:
            ...

    assert type_inspector().attribute_type(my_class, "pt") == Callable[[float], float]


def test_attr_not_there():
    class my_class:
        def pt(self, y: float) -> float:
            ...

    assert type_inspector().attribute_type(my_class, "not_there") is None


def test_iterable_is():
    assert type_inspector().iterable_object(Iterable[int]) == int


def test_iterable_is_not():
    assert type_inspector().iterable_object(int) is None


def test_iterable_nested():
    assert type_inspector().iterable_object(Iterable[Iterable[float]]) == Iterable[float]


def test_callable_not():
    assert type_inspector().callable_type(int) == (None, None)


def test_callable_no_args_method():
    assert type_inspector().callable_type(Callable[[], float]) == ([], float)


def test_callable_args_method():
    assert type_inspector().callable_type(Callable[[int, float], float]) == ([int, float], float)

# TODO: All other types of types that we might have to deal with.
# TODO: make sure that attribute_type throws if you try to add something new to an Iterable[xxx] type! Not add a default value.
# TODO: Make sure other raise NotImplementedError's are ok to remain as such.


@pytest.mark.parametrize("defined_args, actual_args, level, result",
                         [
                             ((float,), (float,), 0, (float,)),
                             ((float,), (Iterable[float],), 1, (float,)),
                             ((float,), (Iterable[Iterable[float]],), 2, (float,)),
                             ((Iterable[float],), (Iterable[float],), 0, (Iterable[float],)),
                             ((Union[int, float],), (float,), 0, (float,)),
                             ((Union[int, float],), (Iterable[float],), 1, (float,)),
                             ((Union[int, float],), (int,), 0, (int,)),
                         ])
def test_find_broadcast_good(defined_args, actual_args, level, result):
    assert type_inspector().find_broadcast_level_for_args(defined_args, actual_args) == (level, result)


@pytest.mark.parametrize("defined_args, actual_args",
                         [
                             ((float,), (int,)),
                             ((float,), (Iterable[int],)),
                             ((float, float), (Iterable[float], float)),
                             ((float,), (int, int)),
                         ])
def test_find_broadcast_bad(defined_args, actual_args):
    assert type_inspector().find_broadcast_level_for_args(defined_args, actual_args) is None
