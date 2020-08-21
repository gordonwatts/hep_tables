from typing import Iterable
from func_adl_xAOD import ServiceXDatasetSource

from hep_tables import xaod_table
from hep_tables.local import _new_make_local

from .conftest import extract_selection, MatchQastleSequence


class Jet:
    def pt(self) -> float:
        ...


class AnEvent:
    def jets(self) -> Iterable[Jet]:
        ...


def test_collect_pts(servicex_ds):
    'Two level test going from xaod_table to a qastle query: integration test'
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f, table_type_info=AnEvent)
    seq = df.jets.pt

    a = _new_make_local(seq)
    assert a is not None
    assert len(a) == 283458

    assert MatchQastleSequence(lambda f: f
                               .Select("lambda e1: e1.jets()")
                               .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
                               ) == extract_selection(servicex_ds)

# TODO: somewhere, test the fact that we can do multiple sources (a list in ServiceXDatasetSource).
# TODO: Should xaod_table demand type info?
