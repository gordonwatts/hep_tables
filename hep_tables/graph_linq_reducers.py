from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

from igraph import Graph, Vertex, Edge  # type:ignore

from hep_tables.graph_info import copy_v_info, e_info, get_e_info, get_v_info, v_info
from hep_tables.transforms import expression_tuple, sequence_downlevel
from hep_tables.util_ast import add_level_to_holder, set_holder_level_index
from hep_tables.util_graph import depth_first_traversal
from hep_tables.utils import QueryVarTracker


def run_linear_reduction(g: Graph, qv: QueryVarTracker):
    '''Reduce a graph to a linear sequence of steps by combining steps that have to run
    in parallel.

    Args:
        g (Graph): [description]
    '''
    max_level = find_highest_level(g)
    for level in range(max_level, 0, -1):
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
        main_seq = [e for e in v.out_edges() if get_e_info(e).main]
        assert len(main_seq) == 1, f'Internal error - only one edge can be labeled main_seq (not {len(main_seq)})'
        main_seq_ast = get_v_info(main_seq[0].target_vertex).node
        new_seq = sequence_downlevel(vs_meta.sequence, qv.new_var_name(), main_seq_ast)
        new_node_dict = {k: add_level_to_holder().visit(v) for k, v in vs_meta.node_as_dict.items()}
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
                    # Track this transform in the tuple
                    vs_meta = get_v_info(v)
                    transforms.append(vs_meta.sequence)
                    for key, val in vs_meta.node_as_dict.items():
                        set_holder_level_index(index).visit(val)
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
        source_vertex (Vertex): The starting point of the edige
        target_vertex (Vertex): The destination of the edge
        e_main (bool): If true make sure that the edge metadata has main marked as true. If false,
                       then don't mark it as true (but if it is already true, then leave it)
    '''
    # See if there already exists an edge:
    old_edge = _find_edge(source_vertex, target_vertex)
    if old_edge is None:
        # Create a vertex
        source_vertex.graph.add_edge(source_vertex, target_vertex, info=e_info(e_main))
    else:
        # See if we need ot do the update.
        if e_main:
            if not old_edge['info'].main:
                old_edge['info'] = e_info(e_main)

