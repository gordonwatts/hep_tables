import ast

from func_adl.util_ast import lambda_build
from hep_tables.utils import QueryVarTracker
from typing import Dict, Iterable, Iterator, List, Optional, OrderedDict, Tuple
from func_adl.object_stream import ObjectStream

from igraph import Graph, Vertex  # type: ignore

from hep_tables.transforms import astIteratorPlaceholder, name_seq_argument, sequence_predicate_base


def build_linq_expression(exp_graph: Graph, qt: QueryVarTracker) -> ObjectStream:
    '''Build a LINQ expression for func_adl from the given the expression as an ast

    Args:
        catalog (ast_sequence_catalog): A full catalog of the ast expressions
    '''
    # Loop over the sequence, generating Select and Where statements
    # at the top level
    build_sequence: Optional[ObjectStream] = None
    ast_dict: Dict[ast.AST, ast.AST] = {}
    for vertices_at_step in depth_first_traversal(exp_graph):
        # For each of the transforms at this step, gather by
        # depth in interation
        t_by_itr_depth = split_by_iteration_depth(vertices_at_step)
        ast_dict_deep: Dict[ast.AST, ast.AST] = {}
        level_holder: Optional[_monad_select_transform] = None
        for level in t_by_itr_depth.keys():
            # For the transforms in this level, put them into a host
            # create a host for this level and fill it
            # Append it to the previous level
            vertices = t_by_itr_depth[level]
            transform_data = [(t['seq'], parent_node_ast(t)) for t in vertices]
            if level_holder is not None:
                np = astIteratorPlaceholder()
                ast_dict[np] = np
                transform_data.append((level_holder, np))

            level_holder = _monad_select_transform(transform_data,
                                                   qt,
                                                   lift_ok=level == 1)

            if len(vertices) > 1:
                for i, t in enumerate(vertices):
                    ast_dict_deep[t['node']] = ast.Subscript(value=astIteratorPlaceholder(), slice=ast.Index(i))
            elif len(vertices) == 1:
                ast_dict_deep[vertices[0]['node']] = astIteratorPlaceholder()

        assert level_holder is not None, 'Internal error: we should never have no statement coming out of this'
        build_sequence = level_holder.sequence(build_sequence, ast_dict)
        ast_dict = ast_dict_deep

        # # And then do the final .sequence on the object
        # if len(transforms) == 1:
        #     t = transforms[0]
        #     build_sequence = t['seq'].sequence(build_sequence, ast_dict)
        #     ast_dict = {
        #         t['node']: astIteratorPlaceholder()
        #     }
        # else:
        #     m_select = _monad_select_transform([(t['seq'], parent_node_ast(t)) for t in transforms], qt)
        #     assert build_sequence is not None
        #     build_sequence = m_select.sequence(build_sequence, ast_dict)
        #     ast_dict = {t['node']: ast.Subscript(value=astIteratorPlaceholder(), slice=ast.Index(i)) for i, t in enumerate(transforms)}

    assert build_sequence is not None
    return build_sequence


def parent_node_ast(v: Vertex) -> Optional[ast.AST]:
    '''Return the parent sequence that this node will use by looking for the properly labeled edge

    Args:
        Vertex ([type]): The vertex we are going to be looking for a parent to.

    Returns:
        ast.AST: AST representing that parent
    '''
    edges = [e for e in v.out_edges() if e['main_seq'] is True]
    if len(edges) == 0:
        return None
    assert len(edges) == 1, f'Internal error - Must be one and only one sequence edge, not {len(edges)}'
    return edges[0].target_vertex['node']


class _monad_select_transform(sequence_predicate_base):
    'A select statement that works on a tuple'
    def __init__(self, tuple_statements: List[Tuple[sequence_predicate_base, Optional[ast.AST]]], qt: QueryVarTracker,
                 lift_ok=False):
        '''Create a select statement that produces a tuple, with each item
        in the tuple being a statement.

        Args:
            tuple_statements (List[Tuple[sequence_predicate_base, ast.AST]): The list of
            statements that we should use, along with the ast's of the sequence they are
            to be built on.
        '''
        self._tuple_statements = tuple_statements
        self._qt = qt
        self._do_lift = lift_ok and len(self._tuple_statements) == 1

    @property
    def lifting(self) -> bool:
        return self._do_lift

    def sequence(self, sequence: Optional[ObjectStream],
                 seq_dict: Dict[ast.AST, ast.AST]) -> ObjectStream:
        # Replace the argument references
        if self._do_lift:
            return self._tuple_statements[0][0].sequence(sequence, seq_dict)
        else:
            assert sequence is not None, 'Internal error: sequence should not be null for a tuple select statement'
            new_name = self._qt.new_var_name()
            arg_replacements = name_seq_argument(seq_dict, new_name)

            # Get the AST's for each of the statements we are going to meld together.
            all_object_streams = [s.sequence(ObjectStream(arg_replacements[a]) if a is not None else None,
                                             arg_replacements)
                                  for s, a in self._tuple_statements]

            # Build them into a tuple that gets called with a massive select. Use a Tuple if we have more than one thing to
            # build.
            if len(all_object_streams) > 1:
                tpl = ast.Tuple(elts=[o._ast for o in all_object_streams])
            else:
                tpl = all_object_streams[0]._ast
            lam = lambda_build(new_name, tpl)
            return sequence.Select(lam)


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
        if all('order' in v.attribute_names() for v in u):
            nodes = tuple(sorted(u, key=lambda v: v['order']))
        else:
            nodes = tuple(u)


def split_by_iteration_depth(vtxs: Iterable[Vertex]) -> OrderedDict[int, List[Vertex]]:
    '''Returns the transform vertices by depth, ordered, starting at the largest and
    going back towards the smallest. If there are gaps, they are filled with empty lists.

    Args:
        vtxs (Iterable[Vertex]): List of the transforms to split

    Returns:
        Dict[int, Iterable[Vertex]]: The transformed split by ordering
    '''
    # find the largest order and back-insert it into the ordered dict.
    max_depth = max(v['itr_depth'] for v in vtxs)
    o: OrderedDict[int, List[Vertex]] = OrderedDict()
    for index in range(max_depth, 0, -1):
        o[index] = []

    for v in vtxs:
        o[v['itr_depth']].append(v)

    return o
