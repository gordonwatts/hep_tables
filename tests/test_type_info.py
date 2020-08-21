from typing import Callable, Iterable
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
        def pt(self, y:float) -> float:
            ...

    assert type_inspector().attribute_type(my_class, "not_there") == None


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
