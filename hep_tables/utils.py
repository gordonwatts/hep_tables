import ast
from typing import Dict, List, Optional, Tuple, Type

from dataframe_expressions import ast_DataFrame


def _find_dataframes(a: ast.AST) -> ast_DataFrame:
    'Find the asts that represent dataframes. Limit to one or failure for now'
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


class QueryVarTracker:
    def __init__(self):
        self._var_name_counter = 1

    def new_var_name(self):
        '''
        Returns the string for a new variable name. Each one is unique.
        '''
        assert self._var_name_counter < 10000
        v = f'e{self._var_name_counter:04}'
        self._var_name_counter += 1
        return v

    def new_term(self, t: Type):
        'Return a new term of type t with a random name'
        from .render import term_info
        return term_info(self.new_var_name(), t)


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
        expr            First hit in the standard ast.NodeVisitor algorithm that is
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
                    result.append(i + 1)
                if c == ')':
                    return result, i + 1
                if c == '(':
                    r, pos = parse_for_commas(part_list[i + 1:])
                    ignore_before = i + pos + 1

        return result, len(part_list)

    commas, _ = parse_for_commas(s[1:-1])
    bounds = [1] + [c + 1 for c in commas] + [len(s)]
    segments = [s[i:j - 1] for i, j in zip(bounds, bounds[1:])]

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


def _is_list(t: Type) -> bool:
    return t.__origin__ is list if not isinstance(t, type) else False  # type: ignore


def _unwrap_list(t: Type) -> Type:
    assert _is_list(t)
    return t.__args__[0]


def _unwrap_if_possible(t: Type) -> Type:
    if _is_list(t):
        return _unwrap_list(t)
    return t


def _same_generic_type(t1: Type, t2: Type) -> bool:
    from typing import _GenericAlias  # type: ignore
    if not isinstance(t1, _GenericAlias) or not isinstance(t2, _GenericAlias):
        return False

    if t1.__origin__ != t2.__origin__:
        return False

    if len(t1.__args__) != len(t2.__args__):
        return False

    return True


def _is_of_type(t1: Type, t2: Type) -> bool:
    '''
    Returns true if t1 is of type t2
    '''
    if t1 == t2:
        return True

    if t2 == object and not _is_list(t1):
        return True

    if not _same_generic_type(t1, t2):
        return False

    for a_t1, a_t2 in zip(t1.__args__, t2.__args__):
        if not _is_of_type(a_t1, a_t2):
            return False

    return True


def _type_replace(source_type: Type, find: Type, replace: Type) -> Optional[Type]:
    '''
    Find `find` as deeply in `source_type` as possible, and replace it with `replace'.

    `_type_replace(List[List[float]], List[object], int) -> List[int]`

    If source_type contains no `find`, then return None
    '''
    from typing import _GenericAlias  # type: ignore
    if isinstance(source_type, _GenericAlias):
        if source_type._name == 'List':
            r = _type_replace(source_type.__args__[0], find, replace)
            if r is not None:
                return List[r]

    if _is_of_type(source_type, find):
        return replace

    return None


def _count_list(t: Type) -> int:
    'Count number of List in a nested List'
    from typing import _GenericAlias  # type: ignore
    if not isinstance(t, _GenericAlias):
        return 0

    if t._name != 'List':
        return 0

    return 1 + _count_list(t.__args__[0])
