import ast
from hep_tables.util_ast import add_level_to_holder
from typing import Dict, Optional

from func_adl.object_stream import ObjectStream
from igraph import Graph

from hep_tables.graph_info import get_e_info, get_v_info
from hep_tables.transforms import sequence_predicate_base
from hep_tables.util_graph import depth_first_traversal, find_main_seq_edge


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
    ast_dict: Dict[ast.AST, ast.AST] = {}
    for vertices_at_step in depth_first_traversal(exp_graph):
        assert len(vertices_at_step) == 1, 'Internal error - only linear execution graphs supported'
        v = vertices_at_step[0]

        v_meta = get_v_info(v)
        main_index: int = 0
        if len(v.out_edges()) > 0:
            main_index = get_e_info(find_main_seq_edge(v)).itr_idx

        seq = v_meta.sequence
        assert isinstance(seq, sequence_predicate_base), 'Internal error - everything should be by now'
        build_sequence = seq.sequence(build_sequence, ast_dict)
        ast_dict = {k: add_level_to_holder(main_index).visit(v)
                    for k, v in v_meta.node_as_dict.items()}

    assert build_sequence is not None
    return build_sequence
