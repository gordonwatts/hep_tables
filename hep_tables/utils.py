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


def _find_root_expr(expr: ast.AST, a: ast.AST) -> ast.AST:
    '''
    Look to see if we can find the root expression for this ast. It will either be `a` or
    it will be an `ast_DataFrame` - return whichever one it is.

    Arguments:
        expr            Expression to find a root
        a               Root

    Result:
        expr            First hit in the standard ast.NodeVistor algorithm that is
                        either the a object or an instance of type `ast_DataFrame`.
    '''
    class root_finder(ast.NodeVisitor):
        def __init__(self, possible_root: ast.AST):
            ast.NodeVisitor.__init__(self)
            self._possible = possible_root
            self.found: Optional[ast.AST] = None

        def visit(self, a: ast.AST):
            if a is self._possible:
                self.found = a
            elif isinstance(a, ast_DataFrame):
                self.found = a
            else:
                ast.NodeVisitor.visit(self, a)

    r = root_finder(a)
    r.visit(expr)
    assert r.found is not None, 'Internal coding error - every expr should have a root'
    return r.found


def _ast_replace(expression: ast.AST, source: ast.AST, dest: ast.AST) -> ast.AST:
    '''
    Scan the tree looking for `source` and replace it with `dest`. No other checking is done.
    '''

    class replace_it(ast.NodeTransformer):
        def __init__(self, source: ast.AST, dest: ast.AST):
            ast.NodeTransformer.__init__(self)
            self._source = source
            self._dest = dest

        def visit(self, a: ast.AST):
            if a is self._source:
                return self._dest

            return ast.NodeTransformer.visit(self, a)

    v = replace_it(source, dest)
    return v.visit(expression)
