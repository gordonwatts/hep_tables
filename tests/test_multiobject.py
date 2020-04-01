from hep_tables import make_local, xaod_table
from .utils_for_testing import f, reduce_wait_time, reset_var_counter # NOQA
from .utils_for_testing import files_back_1, good_transform_request # NOQA
from .utils_for_testing import translate_linq


def test_combine_noop(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.map(lambda j: j).pt
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_simple_capture_and_replace(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.map(lambda j: df).met
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda: e1: (e1, e1)")
        .Select("lambda e3: e3[0].Select(lambda e2: e3[1])")
        .Select("lambda e5: e5.Select(lambda e4: e4.met)")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_object_compare(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.map(lambda j: df.Electrons.DeltaR(j))
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.Electrons(), e1.jets())")
        .Select("lambda e2: e2[0].Select(e3: e2[1].Select(lambda e4: e3.DeltaR(e4))")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt
