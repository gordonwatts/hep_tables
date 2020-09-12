import ast
from typing import Dict, List, Optional, Type, Union, cast

from igraph import Edge, Graph, Vertex  # type: ignore

from hep_tables.util_ast import astIteratorPlaceholder

from .transforms import expression_predicate_base


class v_info:
    '''Information attached to a vertex
    '''
    def __init__(self, level: int, seq: expression_predicate_base, v_type: Type, node: Union[ast.AST, Dict[ast.AST, ast.AST]], order: int = 0):
        '''Create an object to hold the metadata associated with a vertex in our processing graph.

        Args:
            level (int): How many levels of `Iterator` down are we going to process this sequence at?
            seq (sequence_predicate_base): The sequence transform to apply
            type (Type): The output type, relative to the initial type (usually `Iterable[Event]`).
            node (ast.AST): The `ast` that this vertex represents
        '''
        self._level = level
        self._seq = seq
        self._type = v_type
        self._node: Dict[ast.AST, ast.AST] = {node: astIteratorPlaceholder()} if isinstance(node, ast.AST) else node
        self._order = order

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, v_info):
            return False

        if self._level != o._level \
                or self._seq != o._seq \
                or self._type != o._type \
                or self._node != o._node \
                or self._order != o._order:
            return False
        return True

    @property
    def level(self) -> int:
        return self._level

    @property
    def sequence(self) -> expression_predicate_base:
        return self._seq

    @property
    def v_type(self) -> Type:
        '''The type for this vertex. Relative to the initial type.

        Returns:
            Type: Type Type for this, usually `Iterable[X]`.
        '''
        return self._type

    @property
    def node(self) -> ast.AST:
        '''The ast that represents the node this transform handles.

        Returns:
            ast.AST: The `ast` representation
        '''
        assert len(self._node) == 1, 'Internal error: cannot get single node for vertex that represents more than one'
        return list(self._node.items())[0][0]

    @property
    def order(self) -> int:
        '''Returns the order this vertex should be processed in when multiple vertices are being processed.

        Returns:
            int: Integer order. Lower means higher priority
        '''
        return self._order

    @property
    def node_as_dict(self) -> Dict[ast.AST, ast.AST]:
        return self._node


class e_info:
    '''Metadata attached with a vertex Edge.
    '''
    def __init__(self, main_seq: bool, iterator_index: int):
        self._main = main_seq
        self._iterators = iterator_index

    @property
    def main(self) -> bool:
        '''Returns indicator if this edge is meant to be the main sequence (or not)

        Returns:
            bool: True if this edge represents the main sequence.
        '''
        return self._main

    @property
    def itr_idx(self) -> int:
        '''Return the iterator index this edge is using

        Returns:
            int: The iterator index
        '''
        return self._iterators


class g_info:
    '''Metadata associated with the Graph object
    '''
    def __init__(self, global_types: List[Type]):
        '''Initialize the global graph metadata.

        Args:
            global_types (List[Type]): List of types that are active at the global level. They are
            scanned for functions, for example, that can be called at the global level.
        '''
        self._global_types = global_types

    @property
    def global_types(self) -> List[Type]:
        '''List of types that are known at the global level

        Returns:
            List[Type]: List of types known at the global level
        '''
        return self._global_types


def get_v_info(v: Vertex) -> v_info:
    '''Return the vertex metadata attached to the vertex

    Args:
        v (Vertex): The `Vertex` to grab the metadata from

    Returns:
        v_info: The vertex meta-data
    '''
    return cast(v_info, v['info'])


def copy_v_info(old: v_info,
                new_level: Optional[int] = None,
                new_sequence: Optional[expression_predicate_base] = None,
                new_node: Optional[Union[ast.AST, Dict[ast.AST, ast.AST]]] = None):
    new_level = old.level if new_level is None else new_level
    new_seq = old.sequence if new_sequence is None else new_sequence
    new_node = old.node_as_dict if new_node is None else new_node

    new_order = old.order
    new_type = old.v_type

    return v_info(new_level, new_seq, new_type, new_node, new_order)


def get_e_info(e: Edge) -> e_info:
    '''Return the edge metadata.

    Args:
        v (Edge): Edge we should extract the metadata from

    Returns:
        e_info: The Edge metadata
    '''
    return cast(e_info, e['info'])


def get_g_info(g: Graph) -> g_info:
    '''Return the graph metadata

    Args:
        e (Graph): The Graph

    Returns:
        g_info: The Graph metadata
    '''
    return cast(g_info, g['info'])
