import ast
from abc import ABC, abstractmethod

from func_adl.util_ast import lambda_build
from hep_tables.utils import QueryVarTracker
from typing import Dict, List, Optional, Tuple, Union

from dataframe_expressions.utils_ast import CloningNodeTransformer
from func_adl.object_stream import ObjectStream

from hep_tables.hep_table import xaod_table


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
    def sequence(self, sequence: Optional[ObjectStream],
                 seq_dict: Dict[ast.AST, ast.AST]) -> ObjectStream:
        '''Apply the operation to the sequence `sequence`

        Args:
            sequence (ObjectStream): The main `func_adl` sequence that is being
                                     that is being processed and that this predicate
                                     will add to. e.g. `xaod.Select(...)`, and this
                                     predicate will add `.Where(...)`

            seq_dict (Dict[ast.AST, ast.AST]): The `ast` dictionary is keyed by the
            `ast` expression (or `Vertex`). If the transform that is being rendered contains
            a reference to these `ast`'s they should be replaced by the value. This is what allows
            lambda capture in the expressions.

        Returns:
            Dict[ast.AST, ast.AST]: [description]
'''
        ...


class sequence_transform(sequence_predicate_base):
    '''Takes a sequence of type L[T1] and translates it to L[T2].
    '''
    def __init__(self,
                 dependent_asts: List[ast.AST],
                 function: ast.Lambda):
        '''Transforms a sequence with the lambda implied by the `function` argument.

        TODO: Do we care about `dependent_asts`?

        Args:
            dependent_asts (List[ast.AST]):     List of the `ast`'s that must be
                                                replaced in the `function` `ast` during rendering by references to
                                                the stream or monads.

            function (ast.AST):                 The ast that represents a Lambda function (must start with ast.Lambda).
                                                The argument should already be unique.

        Notes:
        '''
        self._function = function
        self._dependent_asts = dependent_asts

    def sequence(self, sequence: Optional[ObjectStream],
                 seq_dict: Dict[ast.AST, ast.AST]) -> ObjectStream:
        '''
        Return a Select statement around the function we are given.
        '''
        # Replace the arguments in the function
        class replace_ast(CloningNodeTransformer):
            def generic_visit(self, node: ast.AST) -> Optional[ast.AST]:
                if node in seq_dict:
                    return seq_dict[node]
                return super().generic_visit(node)

        replaced = replace_ast().visit(self._function)

        # Return the call
        return sequence.Select(replaced)


class sequence_tuple(sequence_predicate_base):
    def __init__(self, transforms: List[Tuple[Union[ast.AST, List[ast.AST]], sequence_predicate_base]]):
        self._trans_info = transforms

    @property
    def transforms(self) -> List[sequence_predicate_base]:
        return [t[1] for t in self._trans_info]

    def sequence(self, sequence: Optional[ObjectStream], seq_dict: Dict[ast.AST, ast.AST]) -> ObjectStream:
        raise NotImplementedError()


class sequence_downlevel(sequence_predicate_base):
    '''Hold onto a transform that has to be processed one level down (a nested select
    statement that allows us to access an array of an array)'''
    def __init__(self, transform: sequence_predicate_base, var_name: str):
        '''Create a transform that will operate on the items in an array that is in the current sequence.

        `b: b.Select(j: transform(j))`

        Args:
            transform (sequence_predicate_base): The transform to operate on the array of array elements in the stream.
            var_name (str): The name of the variable we will use in our Select statement.

        '''
        self._transform = transform
        self._var_name = var_name

    @property
    def transform(self) -> sequence_predicate_base:
        return self._transform

    def sequence(self, sequence: Optional[ObjectStream], seq_dict: Dict[ast.AST, ast.AST]) -> ObjectStream:
        ov = ObjectStream(ast.Name(self._var_name))
        down_level = self.transform.sequence(ov, seq_dict)
        lam = lambda_build(self._var_name, down_level._ast)
        return sequence.Select(lam)


def name_seq_argument(seq_dict: Dict[ast.AST, ast.AST], new_name: str) -> Dict[ast.AST, ast.AST]:
    new_name_ast = ast.Name(id=new_name)

    class replace_arg(CloningNodeTransformer):
        def visit_astIteratorPlaceholder(self, node: astIteratorPlaceholder) -> ast.AST:
            return new_name_ast

    arg_replacements = {k: replace_arg().visit(v) for k, v in seq_dict.items()}
    return arg_replacements


class root_sequence_transform(sequence_predicate_base):
    '''Takes the source of all of this'''
    def __init__(self, ds: xaod_table):
        self._eds = ds

    @property
    def eds(self) -> xaod_table:
        '''Return the xaod_table for this transform

        Returns:
            EventDataset: The root event data source
        '''
        return self._eds

    def sequence(self, sequence: Optional[ObjectStream],
                 seq_dict: Dict[ast.AST, ast.AST]) -> ObjectStream:
        '''For the root of the sequence we return the event stream.

        Args:
            sequence (Optional[ObjectStream]): This is ignored, and should probably be none (or this isn't the root
            of the sequence).
            seq_dict (Dict[ast.AST, ast.AST]): The dictionary of replacements is also ignored.

        Returns:
            Dict[ast.AST, ast.AST]: [description]
        '''
        return self.eds.event_source[0]
        # TODO: Deal with multiple event sources.
