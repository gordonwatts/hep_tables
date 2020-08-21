import ast
from typing import Iterable, Optional, Type

from dataframe_expressions.asts import ast_DataFrame
from func_adl.util_ast import lambda_build
from igraph import Graph, Vertex  # type: ignore

from hep_tables.exceptions import FuncADLTablesException
from hep_tables.hep_table import xaod_table
from hep_tables.transforms import root_sequence_transform, sequence_transform
from hep_tables.type_info import type_inspector
from hep_tables.utils import QueryVarTracker


def ast_to_graph(a: ast.AST, qt: QueryVarTracker,
                 g_in: Optional[Graph] = None,
                 type_system: Optional[type_inspector] = None) -> Graph:
    '''Given an AST from `DataFrame` rendering, build a graph network that
    will work on the `func_adl` backend.

    Args:
        a (ast.AST): The ast from the `render` in `dataframe_expressions`
        g (Graph): The `Graph` under construction, or none if it hasn't been started yet.

    Returns:
        Graph: Returned computational graph
    '''
    g_out = g_in if g_in is not None else Graph(directed=True)
    type_system = type_system if type_system is not None else type_inspector()
    _translate_to_sequence(g_out, type_system, qt).visit(a)
    return g_out


class _translate_to_sequence(ast.NodeVisitor):
    def __init__(self, g: Graph, t_info: type_inspector, qt: QueryVarTracker):
        super().__init__()
        self._g = g
        self._t_inspect = t_info
        self._qt = qt

    def visit_Attribute(self, node: ast.Attribute) -> None:
        '''Processing the `Attribute` ast node. Depending on the context, this is
        probably a function call of some wort.

        - We know this isn't a function call.
        - Get the type from the type system and then act.

        Args:
            node (ast.Attribute): Attribute Node
        '''
        # Make sure the base is already dealt with and in the graph
        self.visit(node.value)

        v_source = _get_vertex_for_ast(self._g, node.value)

        # Now, find a type which has the node on it. This is an implied loop, so we might have to dig
        # through some of the iterable layers to find it.
        v_type = v_source['type']
        attr_type: Optional[Type] = None
        depth = 0
        while attr_type is None:
            v_type = self._t_inspect.iterable_object(v_type)
            depth += 1
            if v_type is None:
                # TODO: This might be an internal error, not a user error. Not totally sure.
                raise FuncADLTablesException(f'Do not know how to apply "{node.attr}" on a sequence that is not iterable ({v_source["type"]})')

            attr_type = self._t_inspect.attribute_type(v_type, node.attr)

        # Next, get out the argument and return type.
        arg_types, return_type = self._t_inspect.callable_type(attr_type)
        if arg_types is None or return_type is None:
            raise FuncADLTablesException(f'Do not know how to deal with an attribute of type {attr_type}')
        if len(arg_types) != 0:
            raise FuncADLTablesException(f'Implied function call to {node.attr} requires {len(arg_types)} arguments - none given.')

        # Get the output type
        seq_out_type = return_type
        for i in range(depth):
            seq_out_type = Iterable[seq_out_type]

        # Code this up as a call, propagating the sequence return type.
        arg_name = self._qt.new_var_name()
        sequence_ph = v_source['node']
        function_call = ast.Call(func=ast.Attribute(value=ast.Name(id=arg_name), attr=node.attr), args=[], keywords=[])
        t = sequence_transform([sequence_ph], lambda_build(arg_name, function_call))

        v = self._g.add_vertex(node=node, seq=t, type=seq_out_type, itr_depth=depth)
        self._g.add_edge(v, v_source, main_seq=True)

    def visit_ast_DataFrame(self, node: ast_DataFrame) -> None:
        '''Visit a root of the tree. This will form the basis of all of the graph.

        Args:
            node (ast_DataFrame): The ast_Dataframe Node

        Notes:

        - There can be no more than one single node at the root of this
        - The node is always at itr depth 1, as it of type `Iterable[Event]`
        '''
        if len(self._g.vs()) != 0:
            raise FuncADLTablesException('func_adl_tables can only handle a single xaod_tables root!')
        df = node.dataframe
        if not isinstance(df, xaod_table):
            raise FuncADLTablesException('func_adl_tables needs an xaod_table as the root')

        self._g.add_vertex(node=node, type=Iterable[df.table_type], seq=root_sequence_transform(df), itr_depth=1)


def _get_vertex_for_ast(g: Graph, node: ast.AST) -> Vertex:
    v_list = list(g.vs.select(lambda v: v['node'] is node))
    assert len(v_list) == 1, f'Internal error: Should be only one node per vertex - found {len(v_list)}'
    return v_list[0]
