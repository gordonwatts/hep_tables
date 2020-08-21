import inspect
from typing import Callable, List, Optional, Tuple, Type


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
    def attribute_type(self, base_type: Type, attribute_name: str) -> Optional[Type]:
        '''Return the type for an attribute.

        Args:
            base_type (type): The class or similar which we are going to be looking up.
            attribute_name (str): The name of the attribute to return

        Returns:
            type: The type of the attribute. If the object has no such attribute, then we return
            None.
        '''
        a = getattr(base_type, attribute_name, None)
        if a is None:
            return None
        if not callable(a):
            raise NotImplementedError()

        # Some sort of method
        sig = inspect.signature(a)
        args: List[type] = [sig.parameters[p].annotation for p in sig.parameters.keys()][1:]
        rtn_type = sig.return_annotation
        return Callable[args, rtn_type]  # type: ignore

    def iterable_object(self, i_type: Type) -> Optional[Type]:
        '''If `i_type` is `Iterable[X]`, then return `x`, otherwise return None.

        Args:
            i_type (type): The type-hint to check

        Returns:
            Optional[type]: Return None if `i_type` isn't iterable. Otherwise return the object.
        '''
        if type(i_type).__name__ != '_GenericAlias':
            return None

        import collections
        if i_type.__origin__ != collections.abc.Iterable:
            return None

        assert len(i_type.__args__) == 1, f'Internal error - iterable with wrong number of args: {i_type}.'
        return i_type.__args__[0]

    def callable_type(self, c_type: Type) -> Tuple[Optional[List[Type]], Optional[Type]]:
        '''Returns information about a callable type, or None

        Args:
            c_type (type): The type-hint for a callable.

        Returns:
            Tuple[Optional[List[type]], type]: Returns `(None, None)` if the type isn't callable, otherwise
            returns a list of argument types and the return type.
        '''
        if type(c_type).__name__ != '_GenericAlias':
            return (None, None)

        import collections
        if c_type.__origin__ != collections.abc.Callable:
            return (None, None)

        return_type = c_type.__args__[-1]
        arg_types = list(c_type.__args__[0:-1])

        return (arg_types, return_type)


    # def sequence_type(self, in_seq_type: Type, transform_info: Tuple[Type, Type]) -> Tuple[Optional[Type], Optional[int]]:
    #     '''Determines the resulting sequence type and how many levels in for a transformation to be applied.

    #     Example:

    #         - in_seq_type=float, transform_info=(float, float). This translates a float to a float, and
    #         no change in depth. Further it operates at the top level. Return: (float, 0).

    #        - in_seq_type=Iterable[float], transform_info=(float, float). This translates a float to a float, and
    #        no change in depth. Further, the transform will operate on items (floats) at one level down. Returns:
    #        (Iterable[float], 1).

    #        - in_seq_type=Iterable[Iterable[Jet]], transform_info(Jet, float). This translates a jet to a float (like
    #        looking at the pt of the jet).It Operates on the jet, so that is two levesl down. Returns: (Iterable[Iterable[float]],
    #        2).

    #        - in_seq_type=Iterable[Iterable[Jet]], transform_info(Iterable[Jet], float). This does some sort of aggregate operation
    #        of a list of jets. It operates on the list of jets, so that is one level down. Returns (Iterable[float], 1).

    #        - in_seq_type=Iterable[float], transform_info=(Jet, float). This is not possible, as the Jet is not in the in_seq_type.
    #        Returns: None

    #     Args:
    #         in_seq_type (Type): The type of the input sequence.
    #         transform_info (Tuple[Type, Type]): Tuple of types that dictate how the transform will apply. The
    #         first type is the type that is consumed, and the second is the output type from the transform. The input
    #         type will be found in the `in_seq_type` and replaced by the output type.

    #     Returns:
    #         Tuple[Type, int]: [description]
    #     '''
    #     raise NotImplementedError()
