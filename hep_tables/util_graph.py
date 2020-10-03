from typing import Iterator, Optional, Tuple, cast

from igraph import Edge, Graph, Vertex  # type: ignore

from hep_tables.graph_info import get_e_info, get_v_info
from hep_tables.util_ast import astIteratorPlaceholder


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
        yield nodes  # type: ignore
        new_nodes = [n.neighbors(mode='in') for n in nodes]
        u = set(n for n_list in new_nodes for n in n_list)
        nodes = tuple(sorted(u, key=lambda v: get_v_info(v).order))


def find_main_seq_edge(v: Vertex) -> Edge:
    '''Look at all out going edges from this vertex and find the one
    marked as a main sequence.

    Throws if no main sequence edge is found.

    Args:
        v (Vertex): The vertex whose edges we should search

    Returns:
        Edge: The main sequence edge
    '''
    main_edges = [e for e in v.out_edges() if get_e_info(e).main]
    assert len(main_edges) == 1, f'Internal error: should be 1 main sequence edge, but there are {len(main_edges)}'
    return main_edges[0]


def get_iterator_index(v: Vertex) -> int:
    '''Return a the iterator index this vertex is going to "generate" - that is, the iterator
    that this vertexes children will be using (output? vertex iterator).

    Args:
        v (Vertex): Vertex on which we should determine the iterator index

    Returns:
        int: The iterator index
    '''
    info = get_v_info(v)
    indices = set((cast(astIteratorPlaceholder, val).iterator_number for _, val in info.node_as_dict.items()))
    assert len(indices) == 1, 'Internal error - too many different iterator indexes - do not know how to process'
    return list(indices)[0]


def parent_iterator_index(v: Vertex) -> int:
    '''Given a vertex, find its main sequence in, and figure out the iterator number for it.
    This is the iterator that the vertex `v` will be iterating over.

    It this is a top level vertex, then use the iterator index it is using.

    Args:
        v (Vertex): The vertex which we will look at its input for.

    Returns:
        int: The index that was used.

    Note:
    TODO: Is this getting used by anyone any longer?
    '''
    if len(v.out_edges()) == 0:
        return get_iterator_index(v)
    else:
        main_edge = find_main_seq_edge(v)
        i = get_e_info(main_edge)
        return i.itr_idx


def child_iterator_in_use(v: Vertex, level: int) -> Optional[int]:
    '''Finds another dependent of v, and returns its iterator number.

    1. It must be to the same level
    1. It must be another already established dependent of v.

    Args:
        v (Vertex): The vertex to search for dependent expressions on.
        level (int): The level which the other dependent must be captured
                     on.

    Returns:
        Optional[int]: None if no dependent vertices at the same level
                       were found, otherwise the iterator index.
    '''
    other_child_edges = v.in_edges()
    if len(other_child_edges) == 0:
        return None

    # Lets can all the edges going out for any with the right level.
    good_level_vert = [
        v
        for v in [get_v_info(e.source_vertex) for e in other_child_edges]
        if v.level == level
    ]
    if len(good_level_vert) == 0:
        return None

    # All vertices are assumed to be using the same level here.
    place_holder_dicts = good_level_vert[0].node_as_dict
    place_holder = list(place_holder_dicts.values())[-1]
    assert isinstance(place_holder, astIteratorPlaceholder)
    return place_holder.iterator_number
