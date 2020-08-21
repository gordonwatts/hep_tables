from typing import Optional, Union, cast, Dict
import ast

from func_adl.object_stream import ObjectStream
from igraph import Graph

from hep_tables.transforms import astIteratorPlaceholder, sequence_predicate_base
from hep_tables.util_graph import depth_first_traversal


def build_linq_expression(exp_graph: Graph) -> ObjectStream:
    '''Build the linq expression from the provided graph. We expect all the hard
    work of combining the graph into a linear sequence and leveld uniformly
    to have been done at this point.

    Args:
        exp_graph:      The collapsted graph
    '''
    # Loop over the sequence, generating Select and Where statements
    # at the top level
    build_sequence: Optional[ObjectStream] = None
    for vertices_at_step in depth_first_traversal(exp_graph):
        assert len(vertices_at_step) == 1, 'Internal error - only linear execution graphs supported'
        v = vertices_at_step[0]

        build_sequence = cast(sequence_predicate_base, v['seq']).sequence(build_sequence, _as_dict(v['node']))

    assert build_sequence is not None
    return build_sequence


def _as_dict(ast_info: Union[ast.AST, Dict[ast.AST, ast.AST]]) -> Dict[ast.AST, ast.AST]:
    '''Make sure that the incoming argument is a dict of ast place holders

    Args:
        ast_info (Union[ast.AST, Dict[ast.AST, ast.AST]]): Argument to be converted into a dict, if needed

    Returns:
        Dict[ast.AST, ast.AST]: A resulting dict with everything needed.
    '''
    if isinstance(ast_info, ast.AST):
        return {ast_info: astIteratorPlaceholder()}
    return ast_info
