import inspect
from typing import Callable, Iterable, List, Optional, Set, Tuple, Type, Union


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
        if i_type.__origin__ != collections.abc.Iterable:  # type: ignore
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
        if c_type.__origin__ != collections.abc.Callable:  # type: ignore
            return (None, None)

        return_type = c_type.__args__[-1]
        arg_types = list(c_type.__args__[0:-1])

        return (arg_types, return_type)

    def _possible_types(self, t: Type) -> Set[Type]:
        '''Explode things like Union

        Args:
            t (Type): The type to examine.

        Returns:
            (type): The list of all types that that this type can represent. If this is `Union[float, int]` return the
            set of the two.
        '''
        if type(t).__name__ == '_GenericAlias':
            if t.__origin__ == Union:
                return set(t.__args__)
        r = set()
        r.add(t)
        return r

    def _compare_simple_types(self, defined: Type, actual: Type):
        '''Determine if two non-deep types are comabible.

        Args:
            defined (Type): The first type of the two to compare
            actual (Type): The second type to compare
        '''
        d_set = self._possible_types(defined)
        a_set = self._possible_types(actual)

        return len(d_set & a_set) > 0

    def find_broadcast_level_for_args(self, defined_args: Iterable[Type], actual: Iterable[Type]) -> Optional[Tuple[int, Iterable[Type]]]:
        '''Return the level to broadcast to and actual arguments if they match an allowed set of arguments.

        Args:
            defined_args (Iterable[Type]): The argument types that are allowed
            actual (Iterable[Type]): The arguments that are provided

        Raises:
            NotImplementedError: [description]

        Returns:
            Optional[Tuple[int, Iterable[Type]]]: None if unwrapping by level is not possible. Otherwise, the level to unwrap to,
            and the types that were actually used (useful if a `Union` type is specified)
        '''
        t_defined = tuple(defined_args)
        t_actual = tuple(actual)
        if len(t_defined) != len(t_actual):
            return None

        # Are the types compatible at this level?
        if all(self._compare_simple_types(d, a) for d, a in zip(t_defined, t_actual)):
            return 0, t_actual

        # We can only go down a level if everyone can go down a level.
        new_actual = [self.iterable_object(a) for a in t_actual]
        if any(a is None for a in new_actual):
            return None

        r = self.find_broadcast_level_for_args(t_defined, new_actual)
        if r is not None:
            return r[0] + 1, r[1]
        return None
