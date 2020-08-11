import ast

from typing import List, Optional
from func_adl.object_stream import ObjectStream
from hep_tables.transforms import astIteratorPlaceholder, sequence_predicate_base, sequence_transform


def test_sequence_predicate_base():
    class mtb(sequence_predicate_base):
        def doit(self):
            pass

        def sequence(self, seq: Optional[ObjectStream]) -> ObjectStream:
            return ObjectStream(ast.Constant(10))

        def args(self) -> List[ast.AST]:
            return []

    mtb()


def test_seq_trans_null(mock_qt):
    sequence_transform([ast.Num(20)], ast.Num(20), mock_qt)


def test_seq_trans_no_args(mock_qt):
    s = sequence_transform([], ast.Num(20), mock_qt)
    base_seq = ObjectStream(ast.Name(id='dude'))
    new_seq = s.sequence(base_seq, {})
    assert ast.dump(new_seq._ast, annotate_fields=False) == "Call(Name('Select', Load()), [Name('dude'), Lambda(arguments([arg('e1000')]), Num(20))], [])"


def test_seq_trans_one_args_no_repl(mock_qt):
    a = ast.Num(10)
    s = sequence_transform([a], ast.Num(20), mock_qt)
    base_seq = ObjectStream(ast.Name(id='dude'))
    new_seq = s.sequence(base_seq, {a: ast.Num(30)})
    assert ast.dump(new_seq._ast, annotate_fields=False) == "Call(Name('Select', Load()), [Name('dude'), Lambda(arguments([arg('e1000')]), Num(20))], [])"


def test_seq_trans_one_args(mock_qt):
    a = ast.Num(10)
    s = sequence_transform([a], a, mock_qt)
    base_seq = ObjectStream(ast.Name(id='dude'))
    new_seq = s.sequence(base_seq, {a: ast.Num(30)})
    assert ast.dump(new_seq._ast, annotate_fields=False) == "Call(Name('Select', Load()), [Name('dude'), Lambda(arguments([arg('e1000')]), Num(30))], [])"


def test_seq_trans_attr(mock_qt):
    a_root: ast.AST = ast.Name(id='e')
    add = sequence_transform([a_root], ast.Attribute(a_root, attr='Jets'), mock_qt)

    base_seq = ObjectStream(ast.Name(id='dude'))

    ref_var = astIteratorPlaceholder()
    s = add.sequence(base_seq, {a_root: ref_var})
    assert ast.dump(s._ast, annotate_fields=False) == \
        "Call(Name('Select', Load()), [Name('dude'), Lambda(arguments([arg('e1000')]), Attribute(Name('e1000'), 'Jets'))], [])"


def test_seq_trans_two_arg(mock_qt):
    a_root_1: ast.AST = ast.Name(id='arg1')
    a_root_2: ast.AST = ast.Name(id='arg2')
    add = sequence_transform([a_root_1, a_root_2],
                             ast.BinOp(op=ast.Add(), left=a_root_1, right=a_root_2),
                             mock_qt)

    base_seq = ObjectStream(ast.Name(id='dude'))

    ref_var = astIteratorPlaceholder()
    s = add.sequence(base_seq, {
        a_root_1: ast.Subscript(value=ref_var, slice=ast.Index(value=ast.Num(0))),
        a_root_2: ast.Subscript(value=ref_var, slice=ast.Index(value=ast.Num(1)))
    })

    assert ast.dump(s._ast, annotate_fields=False) == \
        "Call(Name('Select', Load()), [Name('dude'), Lambda(arguments([arg('e1000')]), BinOp(Subscript(Name('e1000'), Index(Num(0))), Add(), Subscript(Name('e1000'), Index(Num(1)))))], [])"
