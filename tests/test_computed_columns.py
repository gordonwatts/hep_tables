from hep_tables import make_local, xaod_table
from dataframe_expressions import define_alias

from func_adl_xAOD import ServiceXDatasetSource

from servicex import clean_linq

from .conftest import (
    translate_linq,
    extract_selection
    )


def test_simple_column_named_in_second_place(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    jets = df.jets
    jets['ptgev'] = lambda j: j.pt / 1000.0
    seq = df.jets.ptgev
    make_local(seq)

    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt()/1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_simple_column(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    jets = df.jets
    jets['ptgev'] = lambda j: j.pt / 1000.0
    seq = jets.ptgev
    make_local(seq)

    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt()/1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_new_column_in_filter(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    jets = df.jets
    jets['ptgood'] = lambda j: j.pt > 50.0
    seq = jets[jets.ptgood].pt
    make_local(seq)

    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Where(lambda e2: (e2.pt() > 50.0))")
                         .Select("lambda e5: e5.Select(lambda e6: e6.pt())")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_new_column_in_filter_inverted(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    jets = df.jets
    jets['ptgood'] = lambda j: j.pt > 50.0
    seq = jets[~jets.ptgood].pt
    make_local(seq)

    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Where(lambda e2: not (e2.pt() > 50.0))")
                         .Select("lambda e5: e5.Select(lambda e6: e6.pt())")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_make_local_bad(servicex_ds):
    define_alias('', 'ptgev', lambda o: o.pt / 1000.0)
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)

    mc_part = df.TruthParticles('TruthParticles')
    mc_ele = mc_part[(mc_part.pdgId == 11)]

    make_local(mc_ele.ptgev)
    json_1 = clean_linq(extract_selection(servicex_ds))

    make_local(mc_ele.ptgev)
    json_2 = clean_linq(extract_selection(servicex_ds))

    assert json_1 == json_2
