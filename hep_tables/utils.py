import ast
from typing import List, Dict, Optional, Tuple

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


def _find_root_expr(expr: ast.AST, possible_root: ast.AST) -> Optional[ast.AST]:
    '''
    Look to see if we can find the root expression for this ast. It will either be `a` or
    it will be an `ast_DataFrame` - return whichever one it is.

    Arguments:
        expr            Expression to find a root
        possible_root   Root

    Result:
        expr            First hit in the standard ast.NodeVistor algorithm that is
                        either the a object or an instance of type `ast_DataFrame`.

    ## Notes:

    Logic is a bit subtle. Say that `possible_root` is df.jets.

        df.jets.pt                  --> df.jets
        df.eles.pt                  --> df
        sin(df.jets.pt)             --> df.jets
        df.eles.DeltaR(df.jets)     --> df

    '''
    class root_finder(ast.NodeVisitor):
        def __init__(self, possible_root: ast.AST):
            ast.NodeVisitor.__init__(self)
            self._possible = possible_root
            self.found: Optional[ast.AST] = None

        def visit(self, a: ast.AST):
            if a is self._possible:
                if self.found is None:
                    self.found = a
            elif isinstance(a, ast_DataFrame):
                self.found = a
            else:
                ast.NodeVisitor.visit(self, a)

    r = root_finder(possible_root)
    r.visit(expr)
    return r.found


def _parse_elements(s: str) -> List[str]:
    '''
    Return comma separated strings at the top level
    '''
    if s[0] != '(' and s[1] != ')':
        return [s]

    def parse_for_commas(part_list: str) -> Tuple[List[int], int]:
        result = []

        ignore_before = 0
        for i, c in enumerate(part_list):
            if i >= ignore_before:
                if c == ',':
                    result.append(i+1)
                if c == ')':
                    return result, i+1
                if c == '(':
                    r, pos = parse_for_commas(part_list[i+1:])
                    ignore_before = i + pos + 1

        return result, len(part_list)

    commas, _ = parse_for_commas(s[1:-1])
    bounds = [1] + [c + 1 for c in commas] + [len(s)]
    segments = [s[i:j-1] for i, j in zip(bounds, bounds[1:])]

    return segments


def _index_text_tuple(s: str, index: int) -> str:
    '''
    If s is a tuple, then return the index'th item
    '''
    splits = _parse_elements(s)
    if len(splits) == 1:
        return f'{s}[{index}]'

    if len(splits) < index:
        raise Exception(f'Internal Error: attempt to index tuple fail: {s} - index {index}')

    return splits[index]
