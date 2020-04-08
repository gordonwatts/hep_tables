import pytest

from hep_tables import RenderException, curry, make_local, xaod_table

from .utils_for_testing import (
    clean_linq, f, files_back_1, good_transform_request, reduce_wait_time,
    reset_var_counter, translate_linq)


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
    assert clean_linq(json['selection']) == txt


def test_combine_leaf_lambda(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.map(lambda j: j.pt)
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(json['selection']) == txt


def test_combine_leaf_func(good_transform_request, reduce_wait_time, files_back_1):
    def aspt(j):
        return j.pt
    df = xaod_table(f)
    seq = df.jets.map(aspt)
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(json['selection']) == txt


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
    assert clean_linq(json['selection']) == txt


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
    assert clean_linq(json['selection']) == txt


def test_object_compare_curried(good_transform_request, reduce_wait_time, files_back_1):
    @curry
    def c_func(d, j):
        return d.Electrons.DeltaR(j)

    df = xaod_table(f)
    seq = df.jets.map(c_func(df))
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
    assert clean_linq(json['selection']) == txt


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
    assert clean_linq(json['selection']) == txt


def test_two_maps(good_transform_request, reduce_wait_time, files_back_1):
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
    assert clean_linq(json['selection']) == txt


def test_three_maps(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.map(
        lambda j: df.Electrons.map(
            lambda e: df.tracks.map(
                lambda t: t.eta + e.eta + j.eta)))
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.jets(), e1)")
        .Select("lambda e14: e14[0].Select(lambda e3: "
                "e14[1].Electrons()"
                ".Select(lambda e13: e14[1].tracks()"
                ".Select(lambda e2: e2.eta() + e13.eta() + e3.eta())))")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(json['selection']) == txt


def test_four_maps(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.map(
        lambda j: df.Electrons.map(
            lambda e: df.tracks.map(
                lambda t: df.mcs.map(
                    lambda mc: t.eta + e.eta + j.eta + mc.eta()))))
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.jets(), e1)")
        .Select("lambda e14: e14[0].Select("
                "lambda e3: e14[1].Electrons().Select("
                "lambda e13: e14[1].tracks().Select("
                "lambda e2: e14[1].mcs().Select("
                "lambda e5: e2.eta() + e13.eta() + e3.eta() + e5.eta()))))")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(json['selection']) == txt


def test_map_in_filter(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    # MC particles's pt when they are close to a jet.
    jets = df.jets
    mcs = df.mcs
    # This is so ugly: we are doing the mcs.map because we are doing array programming,
    # but that is so far from per-event, which is basically what we want here. This shows
    # up clearly inside the code, unfortunately - as we have to have special workarounds
    # to deal with this.
    near_a_jet = mcs[mcs.map(lambda mc: jets.pt.Count() == 2)]
    make_local(near_a_jet.pt)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.mcs(), e1)")
        .Select("lambda e2: e2[0].Where(lambda e3: e2[1].jets()"
                ".Select(lambda e4: e4.pt()).Count() == 2)")
        .Select('lambda e5: e5.Select(lambda e6: e6.pt())')
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(json['selection']) == txt


def test_map_in_filter_passthrough(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    # MC particles's pt when they are close to a jet.
    mcs = df.mcs
    # This is so ugly: we are doing the mcs.map because we are doing array programming,
    # but that is so far from per-event, which is basically what we want here. This shows
    # up clearly inside the code, unfortunately - as we have to have special workarounds
    # to deal with this.
    near_a_jet = mcs[mcs.map(lambda mc: mc.pt > 10.0)]
    make_local(near_a_jet.pt)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.mcs()")
        .Select("lambda e2: e2.Where(lambda e3: e3.pt() > 10.0)")
        .Select('lambda e5: e5.Select(lambda e6: e6.pt())')
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(json['selection']) == txt


def test_map_with_filter_inside(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    mcs = df.mcs
    jets = df.jets[df.jets.pt > 30]

    pt_total = mcs.map(lambda mc: jets.map(lambda j: 1.0))
    make_local(pt_total)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.mcs(), e1)")
        .Select("lambda e2: e2[0]"
                ".Select(lambda e3: e2[1].jets()"
                ".Where(lambda e4: e4.pt() > 30).Select(lambda e5: 1.0))")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(json['selection']) == txt


def test_map_with_const(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    mcs = df.mcs

    pt_total = mcs.map(lambda mc: 1.0)
    make_local(pt_total)

    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.mcs()")
        .Select("lambda e2: e2.Select(lambda e3: 1.0)")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(json['selection']) == txt


def test_map_in_repeat_root_filter(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    # MC particles's pt when they are close to a jet.
    mcs = df.mcs
    # This is so ugly: we are doing the mcs.map because we are doing array programming,
    # but that is so far from per-event, which is basically what we want here. This shows
    # up clearly inside the code, unfortunately - as we have to have special workarounds
    # to deal with this.
    # This this below should fail - becasue the "mc" is a single particle - so you can't
    # do a count on it!
    seq = mcs[mcs.map(lambda mc: mcs.Count() == 2)].pt
    with pytest.raises(RenderException) as e:
        make_local(seq)

    assert str(e.value).find('list of ojects') != -1
