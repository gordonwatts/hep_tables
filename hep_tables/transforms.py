import ast
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from dataframe_expressions.utils_ast import CloningNodeTransformer
from func_adl.object_stream import ObjectStream
from func_adl.util_ast import lambda_build

from hep_tables.hep_table import xaod_table
from hep_tables.util_ast import astIteratorPlaceholder, reduce_holder_by_level, replace_ast, replace_holder


class expression_predicate_base(ABC):
    '''Base class for all expressions we use to assemble an func_adl expression.
    Everything should derive from this.
    '''
    @abstractmethod
    def render_ast(self, ast_replacements: Dict[ast.AST, ast.AST]) -> ast.AST:
        '''Return the ast we are holding, with replacements done as requested in the dict.

        Args:
            ast_replacements (Dict[ast.AST, ast.AST]): The dictionary of AST replacements.

        Returns:
            (ast.AST): The ast with replacements done. The held ast is not altered in the process.
        '''
        ...


class expression_transform(expression_predicate_base):
    '''A simple expression, function call, etc.
    '''
    def __init__(self, exp: ast.AST):
        '''Initialize with an expression.

        Args:
            exp (ast.AST): The expression this object should hold onto
        '''
        self._exp = exp

    def render_ast(self, ast_replacements: Dict[ast.AST, ast.AST]) -> ast.AST:
        '''Render the AST, doing any replacement required first. The original AST
        is not modified in the process.

        Args:
            ast_replacements (Dict[ast.AST, ast.AST]): ast replacement dict

        Returns:
            ast.AST: New AST with the replacements done.
        '''
        return replace_ast(ast_replacements).visit(self._exp)


class expression_tuple(expression_predicate_base):
    '''A tuple expression - holds several expressions together
    in a tuple.
    '''
    def __init__(self, expressions: List[expression_predicate_base]):
        '''Create a tuple expression - we will run all the expressions at once
        in a python `Tuple`.

        Args:
            expressions (List[expression_predicate_base]): The expressions
        '''
        self._expressions = expressions

    @property
    def transforms(self) -> List[expression_predicate_base]:
        return self._expressions

    def render_ast(self, ast_replacements: Dict[ast.AST, ast.AST]) -> ast.AST:
        '''Render as tuple our expression.

        Args:
            ast_replacements (Dict[ast.AST, ast.AST]): The ast replacements to run on each
            expressions we are holding onto.

        Returns:
            ast.AST: The expressions we are returning
        '''
        rendered_asts = [e.render_ast(ast_replacements) for e in self._expressions]
        return ast.Tuple(elts=rendered_asts)


class sequence_predicate_base(expression_predicate_base):
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


class sequence_downlevel(sequence_predicate_base):
    '''Hold onto a transform that has to be processed one level down (a nested select
    statement that allows us to access an array of an array)'''
    def __init__(self, transform: expression_predicate_base, var_name: str, itr_id: int, main_seq_ast: Optional[ast.AST] = None):
        '''Create a transform that will operate on the items in an array that is in the current sequence.

        `b: b.Select(j: transform(j))`

        Args:
            transform (sequence_predicate_base): The transform to operate on the array of array elements in the stream.
            var_name (str): The name of the variable we will use in our Select statement.
            itr_id (int): The iterator index we are looping over.
            main_seq_ast (ast.AST): The AST that represents the main sequence we are iterating over.

        '''
        self._transform = transform
        self._var_name = var_name
        self._id = itr_id
        self._main_seq = main_seq_ast

    @property
    def transform(self) -> expression_predicate_base:
        return self._transform

    @property
    def sequence_ast(self) -> Optional[ast.AST]:
        return self._main_seq

    @property
    def iterator_idx(self) -> int:
        return self._id

    def sequence(self, sequence: Optional[ObjectStream], seq_dict: Dict[ast.AST, ast.AST]) -> ObjectStream:
        '''Render the sub-expression and run a Select on the item

        Args:
            sequence (Optional[ObjectStream]): The base sequence on which to run
            seq_dict (Dict[ast.AST, ast.AST]): The replacement dictionary to use

        Returns:
            ObjectStream: The new, output, sequence
        '''
        # Render the down-level items - which means going "deep" on the place holder levels.
        downlevel_dict = reduce_holder_by_level(self._id, seq_dict)
        sub_expr = self._transform.render_ast(downlevel_dict)

        select_func = lambda_build(self._var_name, replace_holder(self._id, self._var_name).visit(sub_expr))
        return sequence.Select(select_func)

    def render_ast(self, ast_replacements: Dict[ast.AST, ast.AST]) -> ast.AST:
        '''Do a rendering of the sequence as an expression.

        Args:
            ast_replacements (Dict[ast.AST, ast.AST]): ast replacements

        Returns:
            ast.AST: The ast that represents this downlevel. A place holder will be in
            the select call.
        '''
        assert self._main_seq is not None, 'Internal: Must have a main sequence to properly render_ast'
        seq_repl = replace_ast(ast_replacements).visit(self._main_seq)
        o = ObjectStream(seq_repl)
        rendered_ast = self.sequence(o, ast_replacements)._ast
        return rendered_ast


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

    def render_ast(self, ast_replacements: Dict[ast.AST, ast.AST]) -> ast.AST:
        raise NotImplementedError()
