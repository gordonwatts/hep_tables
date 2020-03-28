import ast
from typing import List, Dict, Optional

from dataframe_expressions import ast_DataFrame


def _find_dataframes(a: ast.AST) -> ast_DataFrame:
    'Find the asts that represent dataframs. Limit to one or failure for now'
    class df_scanner(ast.NodeVisitor):
        def __init__(self):
            self.found_frames: List[ast_DataFrame] = []

        def visit_ast_DataFrame(self, a: ast_DataFrame):
            self.found_frames.append(a)

    scanner = df_scanner()
    scanner.visit(a)
    assert len(scanner.found_frames) > 0, 'All expressions must start with a dataframe'
    assert all(f == scanner.found_frames[0] for f in scanner.found_frames), \
        'Only a single dataframe is supported in any expression'
    return scanner.found_frames[0]


# Counter to help keep variable names unique.
_var_name_counter = 1


def reset_new_var_counter():
    global _var_name_counter
    _var_name_counter = 1


def new_var_name():
    '''
    Returns the string for a new variable name. Each one is unique.
    '''
    global _var_name_counter
    v = f'e{_var_name_counter}'
    _var_name_counter = _var_name_counter + 1
    return v


def to_ast(o: object) -> ast.AST:
    '''
    Convert an object to an ast
    '''
    r = ast.parse(str(o)).body[0]
    assert isinstance(r, ast.Expr)
    return r.value  # NOQA


def to_object(a: ast.AST) -> Optional[object]:
    return ast.literal_eval(a)


def to_args_from_keywords(kws: List[ast.keyword]) -> Dict[str, Optional[object]]:
    '''
    Given keywords return a dict of those ast's converted to something useful.
    '''
    return {k.arg: to_object(k.value) for k in kws if isinstance(k.arg, str)}
