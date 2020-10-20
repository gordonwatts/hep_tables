import types
from typing import Type, TypeVar, Union

# This code came from https://stackoverflow.com/questions/46382170/how-can-i-create-my-own-parameterized-type-in-python-like-optionalt/63964247#63964247
# On how to create a new parameterized type


T = TypeVar('T')


class ConstantMetaMixin:
    type: Type


class ConstantMeta(type):
    cache = {}

    def __class_getitem__(cls: T, key: Type) -> Union[T, Type[ConstantMetaMixin]]:
        if key not in ConstantMeta.cache:
            ConstantMeta.cache[key] = types.new_class(
                f'{cls.__name__}_{key.__name__}',
                (cls,),
                {},
                lambda ns: ns.__setitem__('type', key)
            )

        return ConstantMeta.cache[key]

    def __call__(cls, *args, **kwargs):
        assert getattr(cls, 'type', None) is not None, 'Cannot instantiate Constant generic without parameter'
        return super().__call__(*args, **kwargs)


class Constant(metaclass=ConstantMeta):
    'Constant generic type - parameterize, `Constant[int]`, for example'
    def __init__(self):
        pass

    @staticmethod
    def isconstant(c: Type) -> bool:
        '''Test to see if `c` is a constant type.

        Args:
            c ([type]): Type to check

        Returns:
            bool: True if this guy is a constant
        '''
        return ConstantMeta in getattr(c, "__mro__", [])

    @staticmethod
    def constanttype(c: Type) -> Type:
        '''Return the type of the Constant. If this isn't a constant type, then this will fail.

        Args:
            c (Constant): The Constant[T] type.

        Returns:
            Type: The type we are getting back.
        '''
        return c.type
