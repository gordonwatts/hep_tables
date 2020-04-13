from hep_tables import make_local, xaod_table
from dataframe_expressions import define_alias

from .utils_for_testing import ( # NOQA
    clean_linq, delete_default_downloaded_files, f, files_back_1,
    good_transform_request, reduce_wait_time, reset_var_counter, translate_linq)


def test_simple_column_named_in_second_place(good_transform_request, reduce_wait_time,
                                             files_back_1):
    df = xaod_table(f)
    jets = df.jets
    jets['ptgev'] = lambda j: j.pt / 1000.0
    seq = df.jets.ptgev
    make_local(seq)

    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt()/1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(json['selection']) == txt


def test_simple_column(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    jets = df.jets
    jets['ptgev'] = lambda j: j.pt / 1000.0
    seq = jets.ptgev
    make_local(seq)

    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt()/1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(json['selection']) == txt


def test_new_column_in_filter(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    jets = df.jets
    jets['ptgood'] = lambda j: j.pt > 50.0
    seq = jets[jets.ptgood].pt
    make_local(seq)

    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Where(lambda e2: e2.pt() > 50.0)")
                         .Select("lambda e5: e5.Select(lambda e6: e6.pt())")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(json['selection']) == txt


def test_make_local_bad(good_transform_request, reduce_wait_time, files_back_1):
    define_alias('', 'ptgev', lambda o: o.pt / 1000.0)
    df = xaod_table(f)

    mc_part = df.TruthParticles('TruthParticles')
    mc_ele = mc_part[(mc_part.pdgId == 11)]

    make_local(mc_ele.ptgev)
    json_1 = clean_linq(good_transform_request['selection'])

    make_local(mc_ele.ptgev)
    json_2 = clean_linq(good_transform_request['selection'])

    assert json_1 == json_2
