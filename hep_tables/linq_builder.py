import ast

from func_adl.util_ast import lambda_build
from hep_tables.utils import QueryVarTracker
from typing import Dict, Iterator, List, Optional, Tuple
from func_adl.object_stream import ObjectStream

from igraph import Graph, Vertex  # type: ignore

from hep_tables.transforms import astIteratorPlaceholder, name_seq_argument, sequence_predicate_base, sequence_transform


def depth_first_traversal(g: Graph) -> Iterator[Tuple[Vertex]]:
    '''Generator that will return a sequence of vertices, starting from the "root"

    Notes:
        Why do I have to write this? Isn't this in a library somewhere?
        TODO

    Args:
        g (Graph): The graph we should iterate over

    Returns:
        Tuple[Vertex]: Generator - list of items at each level
    '''
    nodes = tuple(g.vs.select(lambda v: v.degree(mode='out') == 0))
    while len(nodes) != 0:
        yield nodes
        new_nodes = [n.neighbors(mode='in') for n in nodes]
        u = set(n for n_list in new_nodes for n in n_list)
        nodes = tuple(u)


def build_linq_expression(exp_graph: Graph, qt: QueryVarTracker) -> ObjectStream:
    '''Build a LINQ expression for func_adl from the given the expression as an ast

    Args:
        catalog (ast_sequence_catalog): A full catalog of the ast expressions
    '''
    build_sequence: Optional[ObjectStream] = None
    ast_dict: Dict[ast.AST, ast.AST] = {}
    for transforms in depth_first_traversal(exp_graph):
        if len(transforms) == 1:
            t = transforms[0]
            build_sequence = t['seq'].sequence(build_sequence, ast_dict)
            ast_dict = {
                t['node']: astIteratorPlaceholder()
            }
        else:
            m_select = _monad_select_transform([(t['seq'], parent_node_ast(t)) for t in transforms], qt)
            assert build_sequence is not None
            build_sequence = m_select.sequence(build_sequence, ast_dict)
            ast_dict = {t['node']: ast.Subscript(value=astIteratorPlaceholder(), slice=ast.Index(i)) for i, t in enumerate(transforms)}

    assert build_sequence is not None
    return build_sequence


def parent_node_ast(v: Vertex) -> ast.AST:
    '''Return the parent sequence that this node will use by looking for the properly labeled edge

    Args:
        Vertex ([type]): The vertex we are going to be looking for a parent to.

    Returns:
        ast.AST: AST representing that parent
    '''
    edges = [e for e in v.out_edges() if e['main_seq'] is True]
    assert len(edges) == 1, f'Internal error - Must be one and only one sequence edge, not {len(edges)}'
    return edges[0].target_vertex['node']


class _monad_select_transform(sequence_predicate_base):
    'A select statement that works on a tuple'
    def __init__(self, tuple_statements: List[Tuple[sequence_predicate_base, ast.AST]], qt: QueryVarTracker):
        '''Create a select statement that produces a tuple, with each item
        in the tuple being a statement.

        Args:
            tuple_statements (List[Tuple[sequence_predicate_base, ast.AST]): The list of
            statements that we should use, along with the ast's of the sequence they are
            to be built on.
        '''
        self._tuple_statements = tuple_statements
        self._qt = qt

    def sequence(self, sequence: ObjectStream,
                 seq_dict: Dict[ast.AST, ast.AST]) -> ObjectStream:
        # Replace the argument references
        new_name = self._qt.new_var_name()
        arg_replacements = name_seq_argument(seq_dict, new_name)

        # Get the AST's for each of the statements we are going to meld together.
        all_object_streams = [s.sequence(ObjectStream(arg_replacements[a]), arg_replacements) for s, a in self._tuple_statements]

        # Build them into a tuple that gets called with a massive select.
        tpl = ast.Tuple(elts=[o._ast for o in all_object_streams])
        lam = lambda_build(new_name, tpl)
        return sequence.Select(lam)

# def render_as_tree(catalog: ast_sequence_catalog) -> AnyNode:
#     node_lookup: Dict[ast.AST, AnyNode] = {}

#     # Create all the nodes
#     for a, s in catalog.items():
#         node_lookup[a] = AnyNode(node=a, seq=s)

#     # Create the links
#     for a, s in catalog.items():
#         parent = node_lookup[a]
#         for arg_ast in s.args:
#             assert node_lookup[arg_ast].parent is None, 'Internal error, two parent ast node'
#             node_lookup[arg_ast].parent = parent

#     # Find the node with no parent
#     no_parents = [node for _, node in node_lookup.items() if node is None]
#     assert len(no_parents) == 1, 'Internal error, should have a single parent'
#     return no_parents[0]
