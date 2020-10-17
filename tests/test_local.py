from typing import Iterable

from dataframe_expressions import user_func
from func_adl_xAOD import ServiceXDatasetSource

from hep_tables import xaod_table
from hep_tables.local import _new_make_local

from .conftest import MatchQastleSequence, extract_selection

# NOTE:
# These tests are integration tests. Add to them, no problem. If you find a bug,
# however, create a unit test in another test file to make sure that bug
# does not occur. Many integration tests are too complex to really isolate
# a bug or behavior.
#


class Jet:
    def pt(self) -> float:
        ...

    def eta(self) -> float:
        ...

    def phi(self) -> float:
        ...


class Tracks:
    def pt(self) -> float:
        ...

    def eta(self) -> float:
        ...

    def phi(self) -> float:
        ...


class AnEvent:
    def jets(self) -> Iterable[Jet]:
        ...

    def tracks(self) -> Iterable[Tracks]:
        ...

    def met(self) -> float:
        ...


@user_func
def DeltaR(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    '''Dummy routine to emulate the DeltaR function in the backend.
    '''
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


def test_collect_sum_jet(servicex_ds):
    'Two level test going from xaod_table to a qastle query: integration test'

    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f, table_type_info=AnEvent)
    seq = df.jets.pt + df.jets.eta

    a = _new_make_local(seq)
    assert a is not None
    assert len(a) == 283458

    assert MatchQastleSequence(lambda f: f
                               .Select("lambda e1: e1.jets()")
                               .Select("lambda e3: e3.Select(lambda e2: (e2.pt(), e2.eta()))")
                               .Select("lambda e4: e4.Select(lambda e5: (e5[0] + e5[1]))")
                               ) == extract_selection(servicex_ds)


def test_double_map_add(servicex_ds):
    'Two level test going from xaod_table to a qastle query: integration test'
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f, table_type_info=AnEvent)
    jets = df.jets
    tracks = df.tracks

    dr_per_jet = jets.map(lambda j: tracks.map(lambda t: j.pt + t.pt))

    _new_make_local(dr_per_jet)

    assert MatchQastleSequence(lambda f: f
                               .Select("lambda e1: (e1.jets(), e1.tracks())")
                               .Select("lambda e2: (e2[0].Select(lambda j: j.pt()), e2[1].Select(lambda t: t.pt()))")
                               .Select("lambda e4: e4[0].Select(lambda e5: e4[1].Select(lambda e6: e5 + e6))")
                               ) == extract_selection(servicex_ds)


def test_double_map_func(servicex_ds):
    'Two level test going from xaod_table to a qastle query: integration test'
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f, table_type_info=AnEvent)
    jets = df.jets
    tracks = df.tracks

    dr_per_jet = jets.map(lambda j: tracks.map(lambda t: DeltaR(j.eta, j.phi, t.eta, t.phi)))

    _new_make_local(dr_per_jet)

    assert MatchQastleSequence(lambda f: f
                               .Select("lambda e1: (e1.jets(), e1.tracks())")
                               .Select("lambda e2: (e2[0].Select(lambda j: (j.eta(), j.phi())), e2[1].Select(lambda t: (t.eta(), t.phi())))")
                               .Select("lambda e4: e4[0].Select(lambda e5: e4[1].Select(lambda e6: DeltaR(e5[0], e5[1], e6[0], e6[1])))")
                               ) == extract_selection(servicex_ds)


# TODO: somewhere, test the fact that we can do multiple sources (a list in ServiceXDatasetSource).
# TODO: Should xaod_table demand type info?

def test_single_and_double(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f, table_type_info=AnEvent)
    jets = df.jets
    met = df.met

    sum = jets.map(lambda j: j.pt + met)
    # TODO: Make sure there is a test that bombs df.met + df.jet.pt, and the error
    # message is understandable.

    _new_make_local(sum)

    assert MatchQastleSequence(lambda f: f
                               .Select("lambda e1: (e1.jets(), e1.met())")
                               .Select("lambda e2: (e2[0].Select(lambda j: j.pt()), e2[1])")
                               .Select("lambda e4: e4[0].Select(lambda e5: e5 + e4[1])")
                               ) == extract_selection(servicex_ds)
