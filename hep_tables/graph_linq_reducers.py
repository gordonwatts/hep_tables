import ast
from collections import defaultdict
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

from igraph import Edge, Graph, Vertex  # type:ignore

from hep_tables.graph_info import (
    copy_v_info, e_info, get_e_info, get_v_info, v_info)
from hep_tables.transforms import expression_transform, expression_tuple, sequence_downlevel
from hep_tables.util_ast import add_level_to_holder, set_holder_level_index
from hep_tables.util_graph import depth_first_traversal, find_main_seq_edge
from hep_tables.utils import QueryVarTracker


def run_linear_reduction(g: Graph, qv: QueryVarTracker):
    '''Reduce a graph to a linear sequence of steps by combining steps that have to run
    in parallel.

    Args:
        g (Graph): [description]
    '''
    max_level = find_highest_level(g)
    for level in range(max_level, 0, -1):
        reduce_iterator_chaining(g, level, qv)
        reduce_tuple_vertices(g, level, qv)

        if level != 0:
            reduce_level(g, level, qv)


def find_highest_level(g: Graph) -> int:
    '''Finds the highest level vertex in the graph and returns its number. It will thus be 1 to some number.

    Args:
        g (Graph): The graph to find the highest level on

    Returns:
        int: The level equal to the largest level.
    '''
    return max(get_v_info(v).level for v in g.vs())


def reduce_level(g: Graph, level: int, qv: QueryVarTracker):
    '''Find all nodes of level `level` and reduce them by one level to level-1.

    Args:
        g (Graph): The execution graph that is to be reduced
        level (int): All nodes of this level will be reduced by one.
        qv (QueryVarTracker): New variable name generator

    Note:
        It is not possible to reduce a transform that has no inputs.
        There is no way to know the expression that it will be iterating over
        during that reduction!
    '''
    assert level > 0, f'Internal programming error: cannot reduce level {level} - must be 2 or larger'
    for v in (a_good_v for a_good_v in g.vs() if get_v_info(a_good_v).level == level):
        vs_meta = get_v_info(v)
        main_seq_edge = find_main_seq_edge(v)
        main_seq_asts = get_v_info(main_seq_edge.target_vertex).node_as_dict
        main_seq_ast = list(main_seq_asts.keys())[0]
        main_seq_iterator_idx = get_e_info(main_seq_edge).itr_idx
        new_seq = sequence_downlevel(vs_meta.sequence, qv.new_var_name(), main_seq_iterator_idx, main_seq_ast)
        new_node_dict = {k: add_level_to_holder(main_seq_iterator_idx).visit(v) for k, v in vs_meta.node_as_dict.items()}
        v['info'] = copy_v_info(vs_meta, new_sequence=new_seq, new_level=level - 1, new_node=new_node_dict)


def partition_by_parents(vs: List[Vertex]) -> List[List[Vertex]]:
    '''Given a list of vertices, partition them by identical parents.

    Args:
        vs (List[Vertex]): The list of vertices to sort into groups by parents

    Returns:
        List[List[Vertex]]: [description]
    '''
    by_parents = {v: tuple(set(p for p in v.neighbors(mode='out'))) for v in vs}
    organized_vertices: Dict[Tuple[Vertex], List[Vertex]] = defaultdict(list)
    for k, v in by_parents.items():
        organized_vertices[v].append(k)

    return [v_list for v_list in organized_vertices.values()]


def reduce_tuple_vertices(g: Graph, level: int, qv: QueryVarTracker):
    '''Look for places where there are two vertices that need to be executed
    in a single step and combine them into a tuple statement.

    Args:
        g (Graph): The graph to reduce. The graph is modified in place.
    '''
    # Get everything done in steps
    steps = list(depth_first_traversal(g))
    vertices_to_delete = []
    for grouping in steps:
        level_group = [v for v in grouping if get_v_info(v).level == level]
        for p_group in partition_by_parents(level_group):
            if len(p_group) > 1:
                # This is a bit messy because we must take all in-comming and out-going
                # connections and re-hook them up. That plus the fact that in `igraph` vertex
                # and edge pointers are rendered invalid if you delete something from the
                # graph.
                new_vertex = g.add_vertex()

                # Tracking variables to hold onto information until later.
                found_main = False
                transforms = []
                ast_list = {}
                for index, v in enumerate(sorted(p_group, key=lambda k: get_v_info(k).order)):
                    # Get the main sequence first
                    main_edge = find_main_seq_edge(v)

                    # Track this transform in the tuple
                    vs_meta = get_v_info(v)
                    transforms.append(vs_meta.sequence)
                    for key, val in vs_meta.node_as_dict.items():
                        set_holder_level_index(get_e_info(main_edge).itr_idx, index).visit(val)
                        ast_list[key] = val

                    # Update edges to vertices that depend on us
                    dependend_on_us_edges = v.in_edges()
                    for e in dependend_on_us_edges:
                        v_dep = e.source_vertex
                        _update_edge(v_dep, new_vertex, get_e_info(e).main)
                    g.delete_edges(dependend_on_us_edges)

                    # Update edges to vertices we depend on. One trick, make sure we don't over do
                    # the main sequence as when we combine, there will be multiple main sequences bundled into one.
                    dependent_edges = v.out_edges()
                    for e in dependent_edges:
                        v_dep = e.target_vertex
                        is_main = get_e_info(e).main and not found_main
                        _update_edge(new_vertex, v_dep, is_main)
                        found_main = found_main or is_main
                    g.delete_edges(dependent_edges)

                    vertices_to_delete.append(v)

                new_seq = expression_tuple(transforms)
                new_info = v_info(level=level, seq=new_seq, v_type=Any, node=ast_list, order=0)
                new_vertex['info'] = new_info

    # Last thing we do: delete the vertices we no longer need here.
    # Must be done this way b.c. the pointers above become invalid when
    # we do this.
    g.delete_vertices(vertices_to_delete)


def reduce_iterator_chaining(g: Graph, level: int, qt: QueryVarTracker):
    '''Look for multiple iterators coming into a single node and make sure they are reduced to a single, important, iterator,
    by transforming the function to account for the "second" level execution.

    Args:
        g (Graph): Graph to look at
        level (int): the node we should be looking at.
    '''
    for v in (a_good_v for a_good_v in g.vs() if get_v_info(a_good_v).level == level):
        parent_edges = v.out_edges()
        iterator_indices = set(get_e_info(e).itr_idx for e in parent_edges if not get_e_info(e).depth_mark)
        if len(iterator_indices) > 1:
            iterator_indices = iterator_indices - set(get_e_info(e).itr_idx for e in parent_edges if get_e_info(e).main)
            assert len(iterator_indices) > 0, f'Internal error - not enough unique indices: {iterator_indices}'
            seq = get_v_info(v).sequence
            for i in iterator_indices:
                itr_nodes = [e.target_vertex for e in v.out_edges() if get_e_info(e).itr_idx == i]
                parent_asts = list(chain.from_iterable([list(get_v_info(v_parent).node_as_dict) for v_parent in itr_nodes]))
                # Assume all parents of a single iterator have the same path back to that iterator.
                var_name = qt.new_var_name()
                new_expr = seq.render_ast({parent_asts[0]: ast.Name(id=var_name)})
                seq = sequence_downlevel(expression_transform(new_expr), var_name, i, parent_asts[0])
            # Update vertex and edges
            new_v_info = copy_v_info(get_v_info(v), new_sequence=seq)
            v['info'] = new_v_info
            for e in v.out_edges():
                info = get_e_info(e)
                if info.itr_idx in iterator_indices:
                    info.depth_mark = True


def _find_edge(v1: Vertex, v2: Vertex) -> Optional[Edge]:
    '''Find the edge between v1 and v2, and return None if it does not
    exist.

    Args:
        v1 (Vertex): Source vertex
        v2 (Vertex): Dest vertex
    '''
    for e in v1.out_edges():
        if e.target_vertex == v2:
            return e
    return None


def _update_edge(source_vertex: Vertex, target_vertex: Vertex, e_main: bool):
    '''Make sure there is an edge between `source` and `vertex`. Further, mark it as
    the main vertex if hasn't been already.

    Args:
        source_vertex (Vertex): The starting point of the edge
        target_vertex (Vertex): The destination of the edge
        e_main (bool): If true make sure that the edge metadata has main marked as true. If false,
                       then don't mark it as true (but if it is already true, then leave it)
    '''
    # See if there already exists an edge:
    old_edge = _find_edge(source_vertex, target_vertex)
    if old_edge is None:
        # Create a vertex
        # TODO: Make sure the proper iterator number is created here
        source_vertex.graph.add_edge(source_vertex, target_vertex, info=e_info(e_main, 1))
    else:
        # See if we need ot do the update.
        if e_main:
            if not old_edge['info'].main:
                old_edge['info'] = e_info(e_main, get_e_info(old_edge).itr_idx)
