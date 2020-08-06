import ast
from typing import Dict
from hep_tables.transforms import sequence_predicate_base


ast_sequence_catalog = Dict[ast.AST, sequence_predicate_base]

# class ast_sequence_catalog:
#     def save_sequence(self, node: ast.AST, seq: sequence_predicate_base):
#         raise NotImplementedError()

#     def lookup_sequence(self, node: ast.AST) -> sequence_predicate_base:
#         raise NotImplementedError()
