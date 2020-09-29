from hep_tables.graph_info import get_e_info, get_v_info
from igraph import Graph, Vertex, Edge  # type: ignore
from typing import Iterator, Tuple


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


def parent_iterator_index(v_source: Vertex) -> int:
    '''Given a vertex, find its main sequence in, and figure out the iterator number for it.

    Args:
        v_source (Vertex): The vertex which we will look at its input for.

    Returns:
        int: The index that was used.
    '''
    main_edge = find_main_seq_edge(v_source)
    i = get_e_info(main_edge)
    return i.itr_idx
