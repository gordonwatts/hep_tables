from abc import ABC, abstractmethod
import ast
from hep_tables.utils import QueryVarTracker
from typing import List, Dict, Optional, Tuple

from func_adl import EventDataset
from func_adl.object_stream import ObjectStream
from func_adl.util_ast import lambda_build


class astIteratorPlaceholder(ast.AST):
    '''A place holder for the actual variable that references
    the main iterator sequence objects.
    '''
    pass


class sequence_predicate_base(ABC):
    '''Base class that holds a transform that will convert a stream from one form to another.
    '''
    def __init__(self):
        pass

    @abstractmethod
    def sequence(self, sequence: ObjectStream,
                 seq_dict: Dict[ast.AST, ast.AST]) -> ObjectStream:
        '''Apply the operation to the sequence `sequence`

        Args:
            sequence (ObjectStream): The main `func_adl` sequence that is being
                                     that is being processed.

            seq_dict (Dict[ast.AST, ast.AST]): What to use to refer to the sequence.
            The targed of the dicts should contain `astIteratorPlaceholder` objects, which
            are replaced by the proper reference object.
            Mostly used when we are part of a tuple, for accessing parts of the tuple.

        Raises:
            NotImplementedError: [description]

        Returns:
            Dict[ast.AST, ast.AST]: [description]
'''
        raise NotImplementedError()


class sequence_transform(sequence_predicate_base):
    '''Takes a sequence of type L[T1] and translates it to L[T2].
    '''
    def __init__(self,
                 dependent_asts: List[ast.AST],
                 function: ast.AST,
                 qt: QueryVarTracker):
        '''Transforms a sequence with the lambda implied by the `function` argument.

        TODO: Do we care about `dependent_asts`?

        Args:
            dependent_asts (List[ast.AST]): List of the `ast`'s that must be
            replaced in the `function` `ast` during rendering by references to
            the stream or monads.
            function (ast.AST): The `ast` that should build the
            `lambda` function. It contains a number of arguments that must
            be replaced, as listed in the `dependent_asts` argument.

        Notes:
            The `function` will be modifed in place. When it is passed in, do
            not use it for anything else or it might change underneath you!
        '''
        self._function = function
        self._dependent_asts = dependent_asts
        self._qt = qt

        self._called = False

    def sequence(self, sequence: ObjectStream,
                 seq_dict: Dict[ast.AST, ast.AST]) -> ObjectStream:
        '''
        Return a Select statement around the function we are given.
        '''
        # Replace the argument references
        new_name = self._qt.new_var_name()
        arg_replacements = name_seq_argument(seq_dict, new_name)

        # Replace the arguments in the function
        class replace_ast(ast.NodeTransformer):
            def generic_visit(self, node: ast.AST) -> Optional[ast.AST]:
                if node in arg_replacements:
                    return arg_replacements[node]
                return super().generic_visit(node)

        assert not self._called, 'Internal error, due to ast replacement can be called once'
        self._called = True  # Protect against assumption that will bite us miles from here.
        replaced = replace_ast().visit(self._function)

        # Build the lambda
        lb = lambda_build(new_name, replaced)

        # Return the call
        return sequence.Select(lb)


def name_seq_argument(seq_dict: Dict[ast.AST, ast.AST], new_name: str) -> Dict[ast.AST, ast.AST]:
    new_name_ast = ast.Name(id=new_name)

    class replace_arg(ast.NodeTransformer):
        def visit_astIteratorPlaceholder(self, node: astIteratorPlaceholder) -> ast.AST:
            return new_name_ast

    arg_replacements = {k: replace_arg().visit(v) for k, v in seq_dict.items()}
    return arg_replacements


class root_sequence_transform(sequence_predicate_base):
    '''Takes the source of all of this'''
    def __init__(self, ds: EventDataset):
        self._eds = ds

    @property
    def eds(self) -> EventDataset:
        '''Return the EventDataSource for this transform

        Returns:
            EventDataset: The root event data source
        '''
        return self._eds

    def sequence(self, sequence: ObjectStream,
                 seq_dict: Dict[ast.AST, ast.AST]) -> Dict[ast.AST, ast.AST]:
        raise NotImplementedError()
