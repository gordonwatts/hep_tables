from dataframe_expressions import user_func

from hep_tables import make_local, xaod_table

from .utils_for_testing import f, reduce_wait_time, reset_var_counter  # NOQA
from .utils_for_testing import files_back_1, good_transform_request  # NOQA
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


def test_combine_leaf(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.map(lambda j: j.pt)
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
        .Select("lambda e1: (e1.jets(), e1)")
        .Select("lambda e5: e5[0].Select(lambda e3: e5[1])")
        .Select("lambda e6: e6.Select(lambda e4: e4.met())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_object_compare(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.map(lambda j: df.Electrons.DeltaR(j))
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.jets(), e1)")
        .Select('lambda e8: e8[0].Select(lambda e3: '
                'e8[1]'
                '.Electrons()'
                '.Select(lambda e7: e7.DeltaR(e3)))')
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_object_compare_pass_eta(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.map(lambda j: df.Electrons.DeltaR(j.eta))
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.jets(), e1)")
        .Select("lambda e10: e10[0].Select(lambda e3: "
                "e10[1]"
                ".Electrons()"
                ".Select(lambda e9: e9.DeltaR(e3.eta())))")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_two_maps(good_transform_request, reduce_wait_time, files_back_1):

    @user_func
    def DeltaR(e1: float, e2: float) -> float:
        assert False, 'this is a fake function and should never be called'

    df = xaod_table(f)
    seq = df.jets.map(lambda j: df.Electrons.map(lambda e: e.eta + j.eta))
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.jets(), e1)")
        .Select("lambda e14: e14[0].Select(lambda e3: "
                "e14[1]"
                ".Electrons()"
                ".Select(lambda e13: e13.eta() + e3.eta()))")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt
