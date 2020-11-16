import ast
from hep_tables.constant import Constant

from hep_tables.util_graph import child_iterator_in_use, highest_used_order, vertex_iterator_indices
from hep_tables.util_ast import astIteratorPlaceholder
from typing import Any, Dict, Iterable, List, Optional, Type, Union, cast

from dataframe_expressions.asts import ast_Callable, ast_DataFrame, ast_FunctionPlaceholder
from dataframe_expressions.render_dataframe import ast_Filter, render_callable, render_context
from igraph import Graph, Vertex  # type: ignore

from hep_tables.exceptions import FuncADLTablesException
from hep_tables.graph_info import copy_v_info, e_info, g_info, get_g_info, get_v_info, v_info
from hep_tables.hep_table import xaod_table
from hep_tables.transforms import expression_transform, root_sequence_transform
from hep_tables.type_info import type_inspector


def ast_to_graph(a: ast.AST,
                 g_in: Optional[Graph] = None,
                 type_system: Optional[type_inspector] = None,
                 context: Optional[render_context] = None) -> Graph:
    '''Given an AST from `DataFrame` rendering, build a graph network that
    will work on the `func_adl` backend.

    Args:
        a (ast.AST): The ast from the `render` in `dataframe_expressions`
        g_in (Graph): The `Graph` under construction, or none if it hasn't been started yet.
        type_system (type_inspector): Object we can use to examine and resolve needed types
        context (render_context): Object used to help with rendering sub-expressions as we see them.

    Returns:
        Graph: Returned computational graph
    '''
    def create_graph():
        g = Graph(directed=True)
        g['info'] = g_info([])
        return g

    g_out = g_in if g_in is not None else create_graph()
    context = context if context is not None else render_context()
    type_system = type_system if type_system is not None else type_inspector()
    _translate_to_sequence(g_out, type_system, context).visit(a)
    return g_out


class _translate_to_sequence(ast.NodeVisitor):
    def __init__(self, g: Graph, t_info: type_inspector, context: render_context):
        super().__init__()
        self._g = g
        self._t_inspect = t_info
        self._context = context

    def visit(self, node: ast.AST) -> None:
        '''Top level visit. If this ast has already got a node, no need for us to run!

        Args:
            node (ast.AST): Node to process if we can't find it in the graph already
        '''
        if any(self._g.vs.select(lambda v: get_v_info(v).node is node)):
            return

        super().visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        '''Processing the `Attribute` ast node. Depending on the context, this is
        probably a function call of some wort.

        - We know this isn't a function call.
        - Get the type from the type system and then act.

        Args:
            node (ast.Attribute): Attribute Node
        '''
        # Make sure the base is already dealt with and in the graph
        self.visit(node.value)

        v_source = _get_vertex_for_ast(self._g, node.value)
        vs_meta = get_v_info(v_source)

        # Now, find a type which has the node on it. This is an implied loop, so we might have to dig
        # through some of the iterable layers to find it.
        v_type = vs_meta.v_type
        attr_type: Optional[Type] = None
        depth = 0
        while attr_type is None:
            v_type_old = v_type
            v_type = self._t_inspect.iterable_object(v_type)
            depth += 1
            if v_type is None:
                raise FuncADLTablesException(f'Cannot find attribute "{node.attr}" on {v_type_old} (from {vs_meta.v_type})')

            attr_type = self._t_inspect.attribute_type(v_type, node.attr)

        # Next, get out the argument and return type.
        arg_types, return_type = self._t_inspect.callable_type(attr_type)
        if arg_types is None or return_type is None:
            raise FuncADLTablesException(f'Do not know how to deal with an attribute of type {attr_type}')
        if len(arg_types) != 0:
            raise FuncADLTablesException(f'Implied function call to {node.attr} requires {len(arg_types)} arguments - none given.')

        # Get the output type
        seq_out_type = return_type
        for i in range(depth):
            seq_out_type = Iterable[seq_out_type]  # type: ignore

        # Understand if the iterator has changed. By construction (currently) all iterators must be
        # in common coming from a single vertex - so if one has been already used, start with that.
        # This will change when we allow things like jet-chaining.
        already_used_itr = child_iterator_in_use(v_source, depth)
        all_iterator_indices = vertex_iterator_indices(v_source)
        assert len(all_iterator_indices) == 1, 'Internal error: do not deal with number of iterators != 1'
        order = 0
        if already_used_itr is not None:
            itr_index = already_used_itr
            order = highest_used_order(v_source) + 1
        elif depth == vs_meta.level:
            itr_index = all_iterator_indices[0]
        else:
            itr_index = get_g_info(self._g).next_iter_index()

        # Code this up as a call, propagating the sequence return type.
        function_call = ast.Call(func=ast.Attribute(value=node.value, attr=node.attr), args=[], keywords=[])
        t = expression_transform(function_call)

        v = self._g.add_vertex(info=v_info(depth, t, seq_out_type, {node: astIteratorPlaceholder(itr_index)}, order=order))
        self._g.add_edge(v, v_source, info=e_info(True))

    def visit_BinOp(self, node: ast.BinOp) -> None:
        '''Process a python binary operator. We support:

            - Add
            - Sub
            - Mult
            - Div
            - Mod
            - Pow

        Args:
            node (BinOp): AST operator
        '''
        if isinstance(node.op, (ast.FloorDiv, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.MatMult)):
            raise FuncADLTablesException(f'Unsupported binary operator "{node.op}".')

        # Make sure everything below us has a place in the graph
        self.visit(node.left)
        self.visit(node.right)

        left = _get_vertex_for_ast(self._g, node.left)
        left_meta = get_v_info(left)
        right = _get_vertex_for_ast(self._g, node.right)
        right_meta = get_v_info(right)

        # What level will this be operating at?
        func_info = self._t_inspect.find_broadcast_level_for_args((Union[float, int], Union[float, int]),
                                                                  (left_meta.v_type, right_meta.v_type))
        if func_info is None:
            raise FuncADLTablesException(f'Unable to figure out how to {left_meta.v_type} {node.op} {right_meta.v_type}.')

        levels, (l_type, r_type) = func_info
        max_level = max(levels)
        assert max_level == left_meta.level or max_level == right_meta.level, 'TODO: implied loops in binary ops not yet tested'
        level = max_level

        # Figure out the return type given the types of these two
        if (l_type == float) or (r_type == float):
            return_type = float
        elif isinstance(node.op, ast.Div):
            return_type = float
        else:
            return_type = int

        # And build the statement that will do the transform.
        l_func_body = ast.BinOp(left=node.left, op=node.op, right=node.right)
        self._connect_vertices(node, level, return_type, l_func_body, [left, right])

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        '''Process a python Unary operator. We support:

            - +
            - -
            - not

        Args:
            node (UnaryOp): AST operator
        '''
        if isinstance(node.op, ast.Invert):
            raise FuncADLTablesException('Unsupported unary operator: Invert')

        # Make sure everything below us has a place in the graph
        self.visit(node.operand)

        v_o = _get_vertex_for_ast(self._g, node.operand)
        meta_o = get_v_info(v_o)

        # What level will this be operating at?
        if isinstance(node.op, (ast.UAdd, ast.USub)):
            func_info = self._t_inspect.find_broadcast_level_for_args((Union[float, int],),
                                                                      (meta_o.v_type,))
        else:
            func_info = self._t_inspect.find_broadcast_level_for_args((bool,),
                                                                      (meta_o.v_type,))
        if func_info is None:
            raise FuncADLTablesException(f'Unsupported type for unary operator: {node.op} {meta_o.v_type}.')

        levels, (r_type, ) = func_info
        level = max(levels)

        if isinstance(node.op, (ast.UAdd, ast.USub)):
            return_type = r_type
        else:
            return_type = bool

        # And build the statement that will do the transform.
        l_func_body = ast.UnaryOp(op=node.op, operand=node.operand)
        self._connect_vertices(node, level, return_type, l_func_body, [v_o])

    def visit_IfExp(self, node: ast.IfExp) -> None:
        '''Process a python if expression operator.:

        Args:
            node (IfExp): AST operator
        '''
        # Make sure everything below us has a place in the graph
        self.generic_visit(node)

        a_test = node.test
        a_body = node.body
        a_orelse = node.orelse

        v_test = _get_vertex_for_ast(self._g, a_test)
        v_body = _get_vertex_for_ast(self._g, a_body)
        v_orelse = _get_vertex_for_ast(self._g, a_orelse)

        m_test = get_v_info(v_test)
        m_body = get_v_info(v_body)
        m_orelse = get_v_info(v_orelse)

        # What level will this be operating at?
        func_info = self._t_inspect.find_broadcast_level_for_args((bool, Union[float, int, str], Union[float, int, str]),
                                                                  (m_test.v_type, m_body.v_type, m_orelse.v_type))
        if func_info is None:
            raise FuncADLTablesException('Types do not match for if expression. Test must be bool, and body and else must be float, int, or str.')

        levels, (t_test, t_body, t_orelse) = func_info
        level = max(levels)

        # FIgure out the return type
        if (t_body == str) or (t_orelse == str):
            if t_body != t_orelse:
                raise FuncADLTablesException('Both body types must be string in if expression if one is string!')
            return_type = str
        elif (t_body == float) or (t_orelse == float):
            return_type = float
        else:
            return_type = int

        # And build the statement that will do the transform.
        l_func_body = ast.IfExp(test=a_test, body=a_body, orelse=a_orelse)
        self._connect_vertices(node, level, return_type, l_func_body, [v_test, v_body, v_orelse])

    def visit_Compare(self, node: ast.Compare):
        '''Process a python Compare operator. We support:

            - <
            - <=
            - ==
            - !=
            - >
            - >=

        Args:
            node (Compare): AST operator
        '''
        if len(node.ops) != 1:
            raise FuncADLTablesException(f'Unsupported comparison operator - can only work with 2 term comparisons - found {len(node.ops)}.')
        comp_op = node.ops[0]
        if isinstance(comp_op, (ast.Is, ast.IsNot, ast.In, ast.NotIn)):
            raise FuncADLTablesException(f'Unsupported comparison operator "{comp_op}".')

        # Make sure everything below us has a place in the graph
        n_left = node.left
        n_right = node.comparators[0]
        self.visit(n_left)
        self.visit(n_right)

        left = _get_vertex_for_ast(self._g, n_left)
        left_meta = get_v_info(left)
        right = _get_vertex_for_ast(self._g, n_right)
        right_meta = get_v_info(right)

        # What level will this be operating at?
        func_info = self._t_inspect.find_broadcast_level_for_args((Union[float, int], Union[float, int]),
                                                                  (left_meta.v_type, right_meta.v_type))
        if func_info is None:
            raise FuncADLTablesException(f'Operator {comp_op} is unsupported in {left_meta.v_type} {comp_op} {right_meta.v_type}.')

        levels, (l_type, r_type) = func_info
        max_level = max(levels)
        assert max_level == left_meta.level or max_level == right_meta.level, 'TODO: implied loops in binary ops not yet tested'
        level = max_level

        # And build the statement that will do the transform.
        l_func_body = ast.Compare(left=n_left, ops=[comp_op], comparators=[n_right])
        self._connect_vertices(node, level, bool, l_func_body, [left, right])

    def visit_BoolOp(self, node: ast.BoolOp):
        '''Process a python Bool operator. We support:

            - and
            - or

        Args:
            node (Compare): AST operator
        '''
        # Make sure everything below us has a place in the graph
        for n in node.values:
            self.visit(n)

        vertices = [_get_vertex_for_ast(self._g, a) for a in node.values]
        v_meta = [get_v_info(v) for v in vertices]

        # What level will this be operating at?
        func_info = self._t_inspect.find_broadcast_level_for_args(tuple(bool for _ in v_meta),
                                                                  tuple(m.v_type for m in v_meta))
        if func_info is None:
            raise FuncADLTablesException(f'All types are not bool for {node.op} logical operator! {[v.v_type for v in v_meta]}.')

        levels, _ = func_info
        level = max(levels)

        # And build the statement that will do the transform.
        l_func_body = ast.BoolOp(op=node.op, values=list(node.values))
        self._connect_vertices(node, level, bool, l_func_body, vertices)

    def visit_ast_DataFrame(self, node: ast_DataFrame) -> None:
        '''Visit a root of the tree. This will form the basis of all of the graph.

        Args:
            node (ast_DataFrame): The ast_Dataframe Node

        Notes:

        - There can be no more than one single node at the root of this
        - The node is always at itr depth 1, as it of type `Iterable[Event]`
        '''
        if len(self._g.vs()) != 0:
            raise FuncADLTablesException('func_adl_tables can only handle a single xaod_tables root!')
        df = node.dataframe
        if not isinstance(df, xaod_table):
            raise FuncADLTablesException('func_adl_tables needs an xaod_table as the root')

        self._g.add_vertex(info=v_info(0,
                                       root_sequence_transform(df),
                                       Iterable[df.table_type],  # type: ignore
                                       {node: astIteratorPlaceholder(get_g_info(self._g).next_iter_index())}))

    def visit_ast_Filter(self, node: ast_Filter):
        '''We are filtering a sequence here. The filter is evaluated "on the side",
        and the expression is passed through.

        Args:
            node (ast_Filter): The ast node representing the filter
        '''
        self.generic_visit(node)
        a_expr = cast(ast.AST, node.expr)
        a_filter = cast(ast.AST, node.filter)

        v_expr = _get_vertex_for_ast(self._g, a_expr)
        v_filter = _get_vertex_for_ast(self._g, a_filter)
        m_expr = get_v_info(v_expr)
        m_filter = get_v_info(v_filter)

        # Check inputs, get at what level this is operating.
        func_info = self._t_inspect.find_broadcast_level_for_args((bool, Any), (m_filter.v_type, m_expr.v_type))
        if func_info is None:
            raise FuncADLTablesException(f'Where expression must evaluate to type bool, not {m_filter.v_type}')
        levels, (_, return_type) = func_info
        level = levels[0]

        # Hook everything up.
        self._connect_vertices(node, level, return_type, a_expr, [v_expr, v_filter], is_filter=True)

    def visit_Call(self, node: ast.Call) -> None:
        '''Dispatch a call

        - If it is a name, then we had better find it in our type system
        - If it is an attribute, look for anything special first.

        Args:
            node (ast.Call): The ast for the node.

        Raises:
            FuncADLTablesException: If we can't sort out how to call
        '''
        if isinstance(node.func, ast.Name):
            self.visit_Call_Name(node, node.func)
        elif isinstance(node.func, ast.Attribute):
            if hasattr(self, f'visit_Call_{node.func.attr}'):
                m = getattr(self, f'visit_Call_{node.func.attr}')
                m(node.func, node)
            else:
                raise FuncADLTablesException(f'Do not know how to call function "{node.func.attr}".')
        elif isinstance(node.func, ast_FunctionPlaceholder):
            self.visit_Call_ast_FunctionPlaceholder(node.func, node)
        else:
            assert False, f'Internal programming error - a call to {node.func} is not supported.'

    def visit_Call_ast_FunctionPlaceholder(self, func: ast_FunctionPlaceholder, node: ast.Call):
        'Process a function defined as being on the backend in the dataframe_expressions package'
        name = func.callable.__name__
        func_type_info = self._t_inspect.callable_signature(func.callable, False)

        call = ast.Call(func=ast.Name(name), args=node.args, keywords=[])
        self.named_function_call(func_type_info, name, node, call)

    def visit_Call_map(self, attr: ast.Attribute, node: ast.Call) -> None:
        '''We want to map a lambda evaluated now onto the sequence we are currently working on.

        We will use `dataframe_expressions` to render this in context and then evaluate it.

        Args:
            attr (ast.Attribute): The attribute containing the map, including the value it is based on.
            node (ast.Call): The complete callable.
        '''
        # Basic checks
        if len(node.args) != 1:
            raise FuncADLTablesException('Require a single argument, a lambda, to a map call')
        if not isinstance(node.args[0], ast_Callable):
            raise FuncADLTablesException('Require something we can call as an argument to map (like a lambda).')
        callable = cast(ast_Callable, node.args[0])

        # Render the base function, and make sure it is in the graph
        self.visit(attr.value)

        # Run the ast rendering.
        expr, new_context = render_callable(callable, self._context, callable.dataframe)

        # next, we can run the expression
        self._context, old_context = new_context, self._context
        try:
            self.visit(expr)
        finally:
            self._context = old_context

    def visit_Call_Name(self, node: ast.Call, func: ast.Name) -> None:
        '''Process a function call. Use the global type system to figure out what the
        types is/are.

        Args:
            node (ast.Call): [description]
            func (ast.Name): [description]
        '''
        # We have to get the type info for the call
        func_type_info = self._t_inspect.static_function_type(get_g_info(self._g).global_types, func.id)
        if func_type_info is None:
            raise FuncADLTablesException(f'Function "{func.id}" is not defined."')

        self.named_function_call(func_type_info, func.id, node, node)

    def named_function_call(self, function_type_info: Type, func_name: str, node: ast.Call, transform_body: ast.Call):
        '''Given a function name and its signature, code up a call, matching argument
        levels if needed.

        Args:
            function_type_info (Type): The type information for the call: `Callable[[float], int]`
            func_name (str): The name of the function
            node (ast.Call): The ast node this function call represents
            transform_body (ast.Call): The ast to use in the transform - the body of the call.

        '''
        arg_types, return_type = self._t_inspect.callable_type(function_type_info)
        if arg_types is None and return_type is None:
            raise FuncADLTablesException(f'Function "{func_name}" is not defined as being callable!')
        assert arg_types is not None, f'Internal error - a callable function should have at least an empty arg list: {func_name}.'
        if return_type is None:
            raise FuncADLTablesException(f'Function "{func_name}" does not have a defined return type - required.')

        # Evaluate the arguments so we can get their types.
        for a in node.args:
            self.visit(a)
        arg_vtx = [_get_vertex_for_ast(self._g, a) for a in node.args]
        arg_meta = [get_v_info(a) for a in arg_vtx]

        level_type_info = self._t_inspect.find_broadcast_level_for_args(arg_types, [m.v_type for m in arg_meta])
        if level_type_info is None:
            raise FuncADLTablesException(f'Do not know how to call {func_name}({arg_types}) with given argument ({[m.v_type for m in arg_meta]})')
        levels, _ = level_type_info

        # To get the final level, we need to scan through non-constant arguments
        non_const_args = [a for a in zip(arg_meta, levels) if not Constant.isconstant(a[0].v_type)]
        non_const_levels = [a[1] for a in non_const_args]
        level = non_const_levels[0] if len(non_const_levels) > 0 else 0

        # Check that we have no implied loops here
        if not all(non_const_levels[0] == lv for lv in non_const_levels) or any(level != m[0].level for m in non_const_args):
            raise FuncADLTablesException(f'In order to call {func_name}({arg_types}), all items need to have the same number of array dimensions.')

        # Connect everything
        self._connect_vertices(node, level, return_type, transform_body, arg_vtx)

    def _connect_vertices(self, node: ast.AST, level: int, return_type: type,
                          func_body: ast.AST, vertices: List[Vertex], is_filter: bool = False):
        '''Given an expression that is connected with a number of nodes, form the connections. Constants are properly dealt with.

        The main sequence will be the first non-constant vertex given in `vertices.`

        Level will the be the main sequence level's index.

        If all arguments are constants, then the new vertex is marked as a constant.

        And evaluation order will be enforced by the order in `vertices`.

        Args:
            node (ast.AST): The node/ast that this is representing
            level (int): The level that this node should run at
            return_type (type): The type of this node. Converted to `Constant[return_type]` or `Iterable[...return_type]`
                                depending.
            func_body (ast.AST): The ast representing the function body. Put in an `expression_transform`
            vertices (List[Vertex]): List of dependent vertices. Edges will be marked.
            is_filter (bool): Create a filter for this

        Returns:
            [type]: [description]
        '''
        s = expression_transform(func_body, is_filter=is_filter)

        # Go in order to find the first good vertex to give us a level
        # and main sequence.
        non_const_vert = [v for v in vertices if not Constant.isconstant(get_v_info(v).v_type)]
        main_seq = non_const_vert[0] if len(non_const_vert) > 0 else None

        main_dict: Optional[Dict[ast.AST, ast.AST]] = None
        if main_seq is not None:
            main_index = vertex_iterator_indices(main_seq)[0]
            main_dict = {node: astIteratorPlaceholder(main_index)}
        else:
            main_index = 0
            main_dict = {node: node}

        # Fix up the return type
        if main_seq is not None:
            for i in range(level):
                return_type = Iterable[return_type]  # type: ignore
        else:
            return_type = Constant[return_type]  # type: ignore

        # Fix up the source order to make sure we evaluate them in a deterministic
        # order.
        _fixup_vertex_order(vertices)

        # Create the vertex and connect to a and b via edges
        # We make, arbitrarily, the left sequence the main sequence (left is better!)
        op_vertex = self._g.add_vertex(info=v_info(level, s, return_type, main_dict))
        for v in vertices:
            if v in non_const_vert:
                self._g.add_edge(op_vertex, v, info=e_info(v == main_seq))

    def visit_Num(self, node: ast.Num):
        '''Translate a constant into a node for later use.

        Constants are funny - they will be "absorbed" down the line, rather than
        being part of the graph as we go. However, we place them as part of the graph
        so they fit in with the rest of the processing.

        In this sense the level doesn't really matter - so we set it as zero here. But it should
        be ignored. Key is the type - which is Constant[xxx], which marks this as a constant number,
        which is good at any level in subsequent argument resolution.

        Args:
            node (ast.Num): The number
        '''
        i = node.n
        if isinstance(i, float):
            t = float
        elif isinstance(i, int):
            t = int
        else:
            raise FuncADLTablesException(f'Do not know how to deal with a number of type {type(i)}')

        v_i = v_info(0, expression_transform(node), Constant[t], {node: node}, 0)  # type: ignore
        self._g.add_vertex(info=v_i)

    def visit_Str(self, node: ast.Num):
        '''Translate a constant into a node for later use.

        Constants are funny - they will be "absorbed" down the line, rather than
        being part of the graph as we go. However, we place them as part of the graph
        so they fit in with the rest of the processing.

        In this sense the level doesn't really matter - so we set it as zero here. But it should
        be ignored. Key is the type - which is Constant[xxx], which marks this as a constant number,
        which is good at any level in subsequent argument resolution.

        Args:
            node (ast.Num): The number
        '''
        v_i = v_info(0, expression_transform(node), Constant[str], {node: node}, 0)  # type: ignore
        self._g.add_vertex(info=v_i)


def _get_vertex_for_ast(g: Graph, node: ast.AST) -> Vertex:
    v_list = list(g.vs.select(lambda v: get_v_info(v).node is node))
    assert len(v_list) == 1, f'Internal error: Should be only one node per vertex - found {len(v_list)}'
    return v_list[0]


def _fixup_vertex_order(vtxs: List[Vertex]):
    '''The vertices specified here should run in order. Bump the order numbers to make it work.

    Args:
        vtxs (List[Vertex]): Ordered list of vertices
    '''
    order = -1
    for v in vtxs:
        if get_v_info(v).order <= order:
            v['info'] = copy_v_info(get_v_info(v), new_order=order + 1)
        order = get_v_info(v).order
