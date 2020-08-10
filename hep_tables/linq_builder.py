import ast
from typing import Dict, Iterator, List, Optional, Tuple
from func_adl.object_stream import ObjectStream

from igraph import Graph, Vertex  # type: ignore

from hep_tables.transforms import astIteratorPlaceholder, sequence_predicate_base


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


def build_linq_expression(exp_graph: Graph) -> ObjectStream:
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
            m_select = _monad_select_transform([t['seq'] for t in transforms])
            assert build_sequence is not None
            build_sequence = m_select.sequence(build_sequence, ast_dict)

    assert build_sequence is not None
    return build_sequence


class _monad_select_transform(sequence_predicate_base):
    'A select statement that works on a tuple'
    def __init__(self, tuple_statements: List[sequence_predicate_base]):
        '''Create a select statement that produces a tuple, with each item
        in the tuple being a statement.

        Args:
            tuple_statements (List[sequence_predicate_base]): Statement list.
        '''
        self._tuple_statements = tuple_statements

    def sequence(self, sequence: ObjectStream,
                 seq_dict: Dict[ast.AST, ast.AST]) -> ObjectStream:
        raise NotImplementedError()

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
