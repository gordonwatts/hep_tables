import ast
from typing import Dict, List, Optional, Union

from dataframe_expressions.utils_ast import CloningNodeTransformer


class astIteratorPlaceholder(ast.AST):
    _fields = ('itr_idx', 'level_index',)
    '''A place holder for the actual variable that references
    the main iterator sequence objects.

    NOTE: This is a little ugly in the sense that we now have to maintain state information - the possible
    next level down. The issue is that the place we set the next index is far away from the place where we set
    the next level down.
    '''
    def __init__(self, itr_idx: int, level_index: List[Optional[int]] = []):
        '''Iterator placeholder in an expression.

        Args:
            iterator_number (int): The iterator number this placeholder represents.
            level_index (List[Optional[int]], optional): Dereferencing index at a particular level.
                Defaults to [].
        '''
        self.itr_idx = itr_idx
        self.level_index = level_index
        self._new_level = None

    @property
    def new_level(self) -> Optional[int]:
        '''Return the level on deck for this item

        Returns:
            Optional[int]: None if no level index has been specified yet, otherwise the index.
        '''
        return self._new_level

    @property
    def iterator_number(self) -> int:
        '''Return the iterator number this placeholder is tracking.

        Returns:
            int: The iterator number for the sequence.
        '''
        return self.itr_idx

    @property
    def levels(self) -> List[Optional[int]]:
        '''Returns the current set of levels attached with this placeholder.

        Due to the way the system works, the deepest reference is the first. So, if a place holder
        represents a level 2 item, then the first item will be access at level 2.

        Returns:
            List[int]: The tuple access index, by level. With deepest level first. A `None` indicates
            that no indexing should occur at that particular level.
        '''
        return self.level_index

    def set_level_index(self, index: int):
        '''Sets the index for this placeholder at the current level. It is not possible to set more
        than one index per level!

        Args:
            index (int): The index we should set this level at

        '''
        if self._new_level is not None:
            raise Exception('Internal programming error: should never set level twice')
        self._new_level = index

    def next_level(self):
        '''Return a new place holder that represents a level down, making a copy of this placeholder.

        Returns:
            astIteratorPlaceHolder: Contains the new placeholder, with the new level in place.
        '''
        return astIteratorPlaceholder(self.itr_idx, self.levels + [self._new_level])


class replace_holder(CloningNodeTransformer):
    def __init__(self, itr_id: int, v_name: Union[str, ast.AST]):
        super().__init__()
        self._id = itr_id
        if isinstance(v_name, str):
            self._v = ast.Name(id=v_name)
        else:
            self._v = v_name

    def visit_astIteratorPlaceholder(self, node: astIteratorPlaceholder) -> ast.AST:
        if node.iterator_number == self._id:
            if len(node.levels) == 0 or node.levels[-1] is None:
                return self._v
            else:
                return ast.Subscript(value=self._v, slice=ast.Index(value=ast.Num(n=node.levels[-1])))
        else:
            return node


class set_holder_level_index(ast.NodeVisitor):
    def __init__(self, itr_id: int, new_level: int):
        super().__init__()
        self._new_level = new_level
        self._id = itr_id

    def visit_astIteratorPlaceholder(self, node: astIteratorPlaceholder):
        if self._id == node.iterator_number:
            node.set_level_index(self._new_level)


class add_level_to_holder(CloningNodeTransformer):
    def __init__(self, itr_id: int):
        super().__init__()
        self._id = itr_id

    def visit_astIteratorPlaceholder(self, node: astIteratorPlaceholder) -> astIteratorPlaceholder:
        if node.iterator_number == self._id:
            new_level = node.next_level()
            return new_level
        return node


def reduce_holder_by_level(itr_id: int, d: Dict[ast.AST, ast.AST]) -> Dict[ast.AST, ast.AST]:
    '''Replace any `astIteratorPlaceholders` with a version with one less level.

    Args:
        d (Dict[ast.AST, ast.AST]): A dictionary with all k,v pairs. The values will be

    Returns:
        Dict[ast.AST, ast.AST]: New `dict`, with new copies of `ast`'s as necessary.
    '''
    class drop_holder_by_level(CloningNodeTransformer):
        def visit_astIteratorPlaceholder(self, node: astIteratorPlaceholder) -> astIteratorPlaceholder:
            'Drop the level list by one if we can'
            if node.iterator_number != itr_id:
                return node

            if len(node.levels) == 0:
                return node

            return astIteratorPlaceholder(node.iterator_number, node.levels[:-1])

    c = drop_holder_by_level()
    return {k: c.visit(v) for k, v in d.items()}


class replace_ast(CloningNodeTransformer):
    '''Replace any ast we know about in the dict  with a
    ast in the dict.
    '''
    def __init__(self, repl: Dict[ast.AST, ast.AST]):
        '''Create a replacer that will re-build any ast visited by
        it using a replacement found in the dictionary we have of ast's.

        Args:
            repl (Dict[ast.AST, ast.AST]): The dictionary of one ast to be replaced with
            the other.
        '''
        super().__init__()
        self._replace = repl

    def generic_visit(self, node: ast.AST) -> Optional[ast.AST]:
        if node in self._replace:
            return self._replace[node]
        return super().generic_visit(node)
