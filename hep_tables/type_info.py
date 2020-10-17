import inspect
import logging
from typing import Callable, Iterable, List, Optional, Set, Tuple, Type, Union


def _is_method(a) -> bool:
    '''Is a an instance method or not?

    Args:
        a ([type]): The callable type

    Returns:
        bool: True if this represents a method, False otherwise.
    '''
    sig = inspect.signature(a)
    a1 = list(sig.parameters.keys())[0]
    return a1 == 'self'


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
        return self.callable_signature(a, True)

    def static_function_type(self, type_search_list: List[Type], func_name: str) -> Optional[Type]:
        '''Return the type info for a function/attribute attached to a global type.

        Args:
            type_search_list (List[Type]): The set of type contexts (types) searched, in order, for a static
            function.

            func_name (str): The name of the function to search for

        Returns:
            Optional[Type]: None if the function was not found, otherwise return the type of the function.
        '''
        for t in type_search_list:
            a = getattr(t, func_name, None)
            if a is not None:
                if callable(a):
                    if _is_method(a):
                        logging.getLogger(__name__).warning(f'Looking up static function {func_name}, found method on {t}. Ignoring.')
                        return None
                    return self.callable_signature(a, False)
        return None

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

        assert len(i_type.__args__) == 1, f'Internal error - iterable with wrong number of args: {i_type}.'  # type: ignore
        return i_type.__args__[0]  # type: ignore

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

        return_type = c_type.__args__[-1]  # type: ignore
        arg_types = list(c_type.__args__[0:-1])  # type: ignore

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
            if t.__origin__ == Union:  # type: ignore
                return set(t.__args__)  # type: ignore
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

    def find_broadcast_level_for_args(self, defined_args: Iterable[Type], actual: Iterable[Type]) \
            -> Optional[Tuple[Tuple[int], Iterable[Type]]]:
        '''Return the level to broadcast for each argument and
        actual arguments if they match an allowed set of arguments. Return None if we can't
        figure out how to do the broadcast.

        Args:
            defined_args (Iterable[Type]): The argument types that are allowed
            actual (Iterable[Type]): The arguments that are provided

        Returns:
            Optional[Tuple[int, Iterable[Type]]]: None if unwrapping by level is not possible. Otherwise, the level to unwrap to,
            and the types that were actually used (useful if a `Union` type is specified)

        Notes:

            - The rules are a little complex.
              1. Things of matched depth are all ok: (float, float), (Iterable[float] Iterable[float]).
              1. If inputs are off by one, they are also ok: (float, Iterable[float]),
                 (Iterable[float], Iterable[Iterable[float]]).
              1. But unmatched inputs are only ok if they occur at the first level. This is because
                 otherwise the user code gets very hard to understand.
            - TODO: With the `map` function it should be ok to get around this above item, but that isn't
              implemented yet.
        '''
        t_defined = tuple(defined_args)
        t_actual = tuple(actual)
        if len(t_defined) != len(t_actual):
            return None

        # Find out deep we have to go for each argument to find a match
        levels_to_match = [self._level_match(d, a) for d, a in zip(t_defined, t_actual)]
        if any(lv is None for lv in levels_to_match):
            return None

        # Make sure there is a legal difference
        min_level = min(lv[0] for lv in levels_to_match)
        max_level = max(lv[0] for lv in levels_to_match)

        if (max_level - min_level) > 1:
            return None

        if (min_level != max_level) and (min_level > 1):
            return None

        # Return the data
        return tuple(lv[0] for lv in levels_to_match), tuple(lv[1] for lv in levels_to_match)  # type: ignore

    def callable_signature(self, a: Callable, skip_first: bool) -> Type:
        '''Return the Callable[] signature of a method or function

        Args:
            a ([type]): The callable method or function
            skip_first ([bool]): If true, skip the first parameter. Useful
            when dealing with methods vs functions

        Returns:
            Type: [description]
        '''
        sig = inspect.signature(a)
        skip_index = 1 if skip_first else 0
        args: List[type] = [sig.parameters[p].annotation for p in sig.parameters.keys()][skip_index:]
        rtn_type = sig.return_annotation if sig.return_annotation is not inspect.Signature.empty else None
        return Callable[args, rtn_type]  # type: ignore

    def _level_match(self, t_defined: Type, t_actual: Type) -> Optional[Tuple[int, Type]]:
        '''Return how many levels we have to strip iterable off of `t_actual` to find `t_defined`.
        If we can't return a -1.

        Args:
            t_defined (Type): The type that is defined by the method or functions
            t_actual (Type): The type that we are looking at right now

        Returns:
            Tuple[int, Type]:
                level: The level down to access the type
                Type: The actual type given
        '''
        # Are the types compatible at this level?
        if self._compare_simple_types(t_defined, t_actual):
            return 0, t_actual

        # Now see if we can take the type down one an find it.
        a = self.iterable_object(t_actual)
        if a is None:
            return None

        # See if this is good!
        r = self._level_match(t_defined, a)
        if r is None:
            return None
        return r[0] + 1, r[1]
