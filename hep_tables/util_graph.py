from igraph import Graph, Vertex  # type: ignore
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
        if all('order' in v.attribute_names() for v in u):
            nodes = tuple(sorted(u, key=lambda v: v['order']))
        else:
            nodes = tuple(u)
