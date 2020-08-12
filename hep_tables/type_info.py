import inspect
from typing import Callable, List


class type_inspector:
    '''Type info accessor for a particlar type. makes acesss to various
    type information easy.

    The type system is based on the python type system and type shed files

    - The type system for the data model is totally done in python typeshed files
    - These are referenced to understand the types
    - Basic types: int, float
       - Note that float could mean double or similar - python has only one.
    - Use `typing.Iterable` to support a list or sequence of things that doesn't have a known length. For example,
      a list of jets that has been filtered.
       - TODO: Support a sequence when you can take the length directly (after filtering you can't).
    '''
    @staticmethod
    def attribute_type(base_type: type, attribute_name: str) -> type:
        '''Return the type for an attribute.

        Args:
            base_type (type): The class or similar which we are going to be looking up.
            attribute_name (str): The name of the attribute to return

        Returns:
            type: The type of the attribute
        '''
        a = getattr(base_type, attribute_name, None)
        if a is None:
            raise NotImplementedError()
        if not callable(a):
            raise NotImplementedError()

        # Some sort of method
        sig = inspect.signature(a)
        args: List[type] = [sig.parameters[p].annotation for p in sig.parameters.keys()][1:]
        rtn_type = sig.return_annotation
        return Callable[args, rtn_type]  # type: ignore
