from hep_tables.utils import QueryVarTracker
from typing import Dict, List, Tuple
from collections import defaultdict

from igraph import Graph, Vertex  # type:ignore

from hep_tables.transforms import sequence_downlevel, sequence_tuple
from hep_tables.util_graph import depth_first_traversal


def run_linear_reduction(g: Graph, qv: QueryVarTracker):
    '''Reduce a graph to a linear sequence of steps by combining steps that have to run
    in parallel.

    Args:
        g (Graph): [description]
    '''
    max_level = find_highest_level(g)
    for level in range(max_level, 0, -1):
        if level != 1:
            reduce_level(g, level, qv)

        reduce_tuple_vertices(g, level)


def find_highest_level(g: Graph) -> int:
    '''Finds the highest level vertex in the graph and returns its number. It will thus be 1 to some number.

    Args:
        g (Graph): The graph to find the highest level on

    Returns:
        int: The level equal to the largest level.
    '''
    return max(v['itr_depth'] for v in g.vs())


def reduce_level(g: Graph, level: int, qv: QueryVarTracker):
    '''Find all nodes of level `level` and reduce them by one level to level-1.

    Args:
        g (Graph): The execution graph that is to be reduced
        level (int): All nodes of this level will be reduced by one.
        qv (QueryVarTracker): New variable name generator
    '''
    assert level > 1, f'Internal programming error: cannot reduce level {level} - must be 2 or larger'
    for v in (a_good_v for a_good_v in g.vs() if a_good_v['itr_depth'] == level):
        new_seq = sequence_downlevel(v['seq'], qv.new_var_name())
        v['seq'] = new_seq
        v['itr_depth'] = level - 1


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


def reduce_tuple_vertices(g: Graph, level: int):
    '''Look for places where there are two vertices that need to be executed
    in a single step and combine them into a tuple statement.

    Args:
        g (Graph): The graph to reduce. The graph is modified in place.
    '''
    # Get everything done in steps
    steps = list(depth_first_traversal(g))
    vertices_to_delete = []
    for grouping in steps:
        level_group = [v for v in grouping if v['itr_depth'] == level]
        for p_group in partition_by_parents(level_group):
            if len(p_group) > 1:
                transform_pairs = []
                parent_vertices = []
                child_vertices = []
                ast_list = []
                for v in sorted(p_group, key=lambda k: k['order']):
                    transform_pairs.append((v['node'], v['seq']))
                    ast_list.append(v['node'])

                    # Delete the edges from this vertex into the graph, and replace them with the new ones
                    children = v.neighbors(mode='in')
                    child_vertices += list(children)
                    g.delete_edges([(p, v) for p in children])

                    parents = v.neighbors(mode='out')
                    parent_vertices += list(parents)
                    g.delete_edges([(v, p) for p in parents])

                    vertices_to_delete.append(v)

                new_seq = sequence_tuple(transform_pairs)
                new_vertex = g.add_vertex(itr_depth=level, node=ast_list, seq=new_seq, order=0)
                g.add_edges([(new_vertex, p) for p in set(parent_vertices)])
                g.add_edges([(p, new_vertex) for p in set(child_vertices)])

    # Last thing we do: delete the vertices we no longer need here.
    # Must be done this way b.c. the pointers above become invalid when
    # we do this.
    g.delete_vertices(vertices_to_delete)
