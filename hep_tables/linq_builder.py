import ast
from typing import Dict, Optional
from hep_tables.catalog import ast_sequence_catalog
from hep_tables.transforms import root_sequence_transform
from func_adl import ObjectStream
from anytree import AnyNode, LevelGroupOrderIter


def build_linq_expression(tree: AnyNode) -> ObjectStream:
    '''Build a LINQ expression for func_adl from the given the expression as an ast

    Args:
        catalog (ast_sequence_catalog): A full catalog of the ast expressions
    '''
    # Loop over every level, accumulating the items
    built_seq: Optional[ObjectStream] = None
    for transforms in LevelGroupOrderIter(tree):
        assert len(transforms) == 1, 'More is not implemented yet'
        t = transforms[0]
        built_seq = t.seq.sequence(built_seq)

    assert built_seq is not None, 'Internal error - no sequence built'
    return built_seq


def render_as_tree(catalog: ast_sequence_catalog) -> AnyNode:
    node_lookup: Dict[ast.AST, AnyNode] = {}

    # Create all the nodes
    for a, s in catalog.items():
        node_lookup[a] = AnyNode(node=a, seq=s)

    # Create the links
    for a, s in catalog.items():
        parent = node_lookup[a]
        for arg_ast in s.args:
            assert node_lookup[arg_ast].parent is None, 'Internal error, two parent ast node'
            node_lookup[arg_ast].parent = parent

    # Find the node with no parent
    no_parents = [node for _, node in node_lookup.items() if node is None]
    assert len(no_parents) == 1, 'Internal error, should have a single parent'
    return no_parents[0]
