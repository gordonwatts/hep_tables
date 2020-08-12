

from typing import Callable
from hep_tables.type_info import type_inspector


def test_attr_simple():

    class my_class:
        def pt(self) -> float:
            ...

    assert type_inspector.attribute_type(my_class, "pt") == Callable[[], float]


def test_attr_one_arg():
    class my_class:
        def pt(self, y: float) -> float:
            ...

    assert type_inspector.attribute_type(my_class, "pt") == Callable[[float], float]
