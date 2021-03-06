import pytest

from hep_tables import RenderException, curry, make_local, xaod_table

from dataframe_expressions import user_func

from func_adl_xAOD import ServiceXDatasetSource

from servicex import clean_linq

from .conftest import (
    translate_linq,
    extract_selection
    )


def test_combine_noop(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.map(lambda j: j).pt
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_combine_leaf_lambda(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.map(lambda j: j.pt)
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_combine_leaf_func(servicex_ds):
    def aspt(j):
        return j.pt
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.map(aspt)
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_simple_capture_and_replace(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.map(lambda j: df).met
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.jets(), e1)")
        .Select("lambda e5: e5[0].Select(lambda e3: e5[1])")
        .Select("lambda e6: e6.Select(lambda e4: e4.met())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_object_compare(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.map(lambda j: df.Electrons.DeltaR(j))
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.jets(), e1)")
        .Select('lambda e8: e8[0].Select(lambda e3: '
                'e8[1]'
                '.Electrons()'
                '.Select(lambda e7: e7.DeltaR(e3)))')
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_object_compare_curried(servicex_ds):
    @curry
    def c_func(d, j):
        return d.Electrons.DeltaR(j)

    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.map(c_func(df))
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.jets(), e1)")
        .Select('lambda e8: e8[0].Select(lambda e3: '
                'e8[1]'
                '.Electrons()'
                '.Select(lambda e7: e7.DeltaR(e3)))')
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_object_compare_pass_eta(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.map(lambda j: df.Electrons.DeltaR(j.eta))
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.jets(), e1)")
        .Select("lambda e10: e10[0].Select(lambda e3: "
                "e10[1]"
                ".Electrons()"
                ".Select(lambda e9: e9.DeltaR(e3.eta())))")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_two_maps(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.map(lambda j: df.Electrons.map(lambda e: e.eta + j.eta))
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.jets(), e1)")
        .Select("lambda e14: e14[0].Select(lambda e3: "
                "e14[1]"
                ".Electrons()"
                ".Select(lambda e13: e13.eta() + e3.eta()))")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_three_maps(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.map(
        lambda j: df.Electrons.map(
            lambda e: df.tracks.map(
                lambda t: t.eta + e.eta + j.eta)))
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.jets(), e1)")
        .Select("lambda e14: e14[0].Select(lambda e3: "
                "e14[1].Electrons()"
                ".Select(lambda e13: e14[1].tracks()"
                ".Select(lambda e2: e2.eta() + e13.eta() + e3.eta())))")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_four_maps(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.map(
        lambda j: df.Electrons.map(
            lambda e: df.tracks.map(
                lambda t: df.mcs.map(
                    lambda mc: t.eta + e.eta + j.eta + mc.eta()))))
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.jets(), e1)")
        .Select("lambda e14: e14[0].Select("
                "lambda e3: e14[1].Electrons().Select("
                "lambda e13: e14[1].tracks().Select("
                "lambda e2: e14[1].mcs().Select("
                "lambda e5: e2.eta() + e13.eta() + e3.eta() + e5.eta()))))")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_map_in_filter(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
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
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.mcs(), e1)")
        .Select("lambda e2: e2[0].Where(lambda e3: e2[1].jets()"
                ".Select(lambda e4: e4.pt()).Count() == 2)")
        .Select('lambda e5: e5.Select(lambda e6: e6.pt())')
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_map_in_filter_passthrough(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    # MC particles's pt when they are close to a jet.
    mcs = df.mcs
    # This is so ugly: we are doing the mcs.map because we are doing array programming,
    # but that is so far from per-event, which is basically what we want here. This shows
    # up clearly inside the code, unfortunately - as we have to have special workarounds
    # to deal with this.
    near_a_jet = mcs[mcs.map(lambda mc: mc.pt > 10.0)]
    make_local(near_a_jet.pt)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.mcs()")
        .Select("lambda e2: e2.Where(lambda e3: e3.pt() > 10.0)")
        .Select('lambda e5: e5.Select(lambda e6: e6.pt())')
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_map_with_filter_inside(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    mcs = df.mcs
    jets = df.jets[df.jets.pt > 30]

    pt_total = mcs.map(lambda mc: jets.map(lambda j: 1.0))
    make_local(pt_total)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.mcs(), e1)")
        .Select("lambda e2: e2[0]"
                ".Select(lambda e3: e2[1].jets()"
                ".Where(lambda e4: e4.pt() > 30).Select(lambda e5: 1.0))")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_map_with_2filters_inside_twice(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)

    eles = df.Electrons('Electrons')
    mc_part = df.TruthParticles('TruthParticles')
    mc_ele = mc_part[mc_part.pdgId == 11]
    good_mc_ele = mc_ele[mc_ele.ptgev > 20]

    ele_mcs = eles.map(lambda reco_e: good_mc_ele)

    make_local(ele_mcs)
    json_1 = clean_linq(extract_selection(servicex_ds))
    make_local(ele_mcs)
    json_2 = clean_linq(extract_selection(servicex_ds))

    assert json_1 == json_2


def test_map_statement_output_format(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)

    eles = df.tracks
    mc_part = df.mcs

    good_eles = eles[eles.pt > 20]

    ele_mcs = good_eles.map(lambda e: mc_part)
    make_local(ele_mcs.pt)

    json = clean_linq(extract_selection(servicex_ds))
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.tracks(), e1)")
        .Select("lambda e6: (e6[0].Where(lambda e7: e7.pt() > 20), e6[1])")
        .Select("lambda e2: e2[0].Select(lambda e3: e2[1].mcs())")
        .Select("lambda e4: e4.Select(lambda e5: e5.Select(lambda e8: e8.pt()))")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json == txt


def test_map_with_filter_inside_call(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    mcs = df.mcs
    jets = df.jets()[df.jets().pt > 30]

    pt_total = mcs.map(lambda mc: jets.map(lambda j: 1.0))
    make_local(pt_total)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.mcs(), e1)")
        .Select("lambda e2: e2[0]"
                ".Select(lambda e3: e2[1].jets()"
                ".Where(lambda e4: e4.pt() > 30).Select(lambda e5: 1.0))")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_map_with_const(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    mcs = df.mcs

    pt_total = mcs.map(lambda mc: 1.0)
    make_local(pt_total)

    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.mcs()")
        .Select("lambda e2: e2.Select(lambda e3: 1.0)")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_map_with_count(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    mcs = df.mcs
    jets = df.jets

    seq = mcs.map(lambda mc: jets.map(lambda j: j.pt + mc.pt).Count())
    make_local(seq)

    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.mcs(), e1)")
        .Select("lambda e2: e2[0].Select(lambda e3: e2[1].jets()"
                ".Select(lambda e4: e4.pt() + e3.pt()).Count())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_seq_map_with_count(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    mcs = df.mcs
    jets = df.jets

    b = mcs.map(lambda mc: jets.map(lambda j: j.pt + mc.pt))
    seq = b.map(lambda p: p.Count())
    make_local(seq)

    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.mcs(), e1)")
        .Select("lambda e2: e2[0].Select(lambda e3: e2[1].jets()"
                ".Select(lambda e4: e4.pt() + e3.pt()))")
        .Select("lambda e5: e5.Select(lambda e6: e6.Count())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_map_user_func_is_iterator(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    mcs = df.mcs
    jets = df.jets

    from dataframe_expressions import user_func
    @user_func
    def DeltaR(p1_eta: float) -> float:
        assert False

    mcs['jets_pt'] = lambda mc: jets.map(lambda j: DeltaR(mc.phi))
    make_local(mcs.jets_pt.Count())

    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.mcs(), e1)")
        .Select("lambda e2: e2[0].Select(lambda e3: e2[1].jets()"
                ".Select(lambda e4: DeltaR(e3.phi())))")
        .Select("lambda e5: e5.Select(lambda e6: e6.Count())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_map_in_repeat_root_filter(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    # MC particles's pt when they are close to a jet.
    mcs = df.mcs
    # This is so ugly: we are doing the mcs.map because we are doing array programming,
    # but that is so far from per-event, which is basically what we want here. This shows
    # up clearly inside the code, unfortunately - as we have to have special workarounds
    # to deal with this.
    # This this below should fail - because the "mc" is a single particle - so you can't
    # do a count on it!
    seq = mcs[mcs.map(lambda mc: mcs.Count() == 2)].pt
    with pytest.raises(RenderException) as e:
        make_local(seq)

    assert str(e.value).find('requires as input') != -1


def test_capture_inside_with_call(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.map(lambda j: df.Electrons().Count())
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.jets(), e1)")
        .Select("lambda e14: e14[0].Select(lambda e3: "
                "e14[1]"
                ".Electrons()"
                ".Count())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_count_of_sequence_inside_filter_2maps(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    mc_part = df.TruthParticles('TruthParticles')
    eles = df.Electrons('Electrons')

    eles['near_mcs'] = lambda reco_e: mc_part
    eles['hasMC'] = lambda e: e.near_mcs.Count() > 0

    make_local(eles[eles.hasMC].pt)

    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.Electrons('Electrons'), e1)")
        .Select("lambda e2: e2[0].Where(lambda e3: "
                "e2[1]"
                ".TruthParticles('TruthParticles')"
                ".Count() > 0)")
        .Select("lambda e4: e4.Select(lambda e5: e5.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_function_return_type_with_maps(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)

    mc_part = df.TruthParticles('TruthParticles')
    eles = df.Electrons('Electrons')

    from dataframe_expressions import user_func
    @user_func
    def DeltaR(p1_eta: float) -> float:
        assert False

    @curry
    def near(mcs, e):
        'Return all particles in mcs that are DR less than 0.5'
        return mcs[lambda m: DeltaR(e.eta()) < 0.5]

    # This gives us a list of events, and in each event, good electrons,
    # and then for each good electron, all good MC electrons that are near by
    ele_mcs = eles.map(near(mc_part))
    make_local(ele_mcs.map(lambda e: e.Count()))

    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.Electrons('Electrons'), e1)")
        .Select("lambda e2: e2[0].Select(lambda e3: "
                "e2[1]"
                ".TruthParticles('TruthParticles')"
                ".Where(lambda e6: DeltaR(e3.eta()) < 0.5))")
        .Select("lambda e4: e4.Select(lambda e5: e5.Count())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_multi_object_monads(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)

    mc_part = df.TruthParticles('TruthParticles')
    eles = df.Electrons('Electrons')

    from dataframe_expressions import user_func
    @user_func
    def DeltaR(p1_eta: float) -> float:
        assert False

    def near(mcs, e):
        'Return all particles in mcs that are DR less than 0.5'
        return mcs[lambda m: DeltaR(e.eta()) < 0.5]

    # This gives us a list of events, and in each event, good electrons,
    # and then for each good electron, all good MC electrons that are near by
    eles['near_mcs'] = lambda reco_e: near(mc_part, reco_e)
    eles['hasMC'] = lambda e: e.near_mcs.Count() > 0

    make_local(eles[eles.hasMC].pt)

    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.Electrons('Electrons'), e1)")
        .Select("lambda e2: e2[0].Where(lambda e3: "
                "e2[1]"
                ".TruthParticles('TruthParticles')"
                ".Where(lambda e6: DeltaR(e3.eta()) < 0.5).Count() > 0)")
        .Select("lambda e4: e4.Select(lambda e5: e5.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_multi_object_call_with_same_thing_twice(servicex_ds):
    # df.Electrons appears inside a call that has unwrapped the sequence.
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)

    mc_part = df.TruthParticles('TruthParticles')
    eles = df.Electrons('Electrons')

    # This gives us a list of events, and in each event, good electrons, and then for each
    # good electron, all good MC electrons that are near by
    eles['near_mcs'] = lambda reco_e: mc_part
    eles['hasMC'] = lambda e: e.near_mcs.Count() > 0

    make_local(eles[~eles.hasMC].pt)

    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.Electrons('Electrons'), e1)")
        .Select("lambda e2: e2[0].Where(lambda e3: "
                "not e2[1]"
                ".TruthParticles('TruthParticles')"
                ".Count() > 0)")
        .Select("lambda e4: e4.Select(lambda e5: e5.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_reference_unfiltered_by_filtered(servicex_ds):
    # df.Electrons appears inside a call that has unwrapped the sequence.
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)

    @user_func
    def DeltaR(p1_eta: float) -> float:
        assert False, 'This should never be called'

    mc_part = df.TruthParticles('TruthParticles')
    eles = df.Electrons('Electrons')

    def dr(e, mc):
        return DeltaR(e.eta())

    def very_near2(mcs, e):
        'Return all particles in mcs that are DR less than 0.5'
        return mcs[lambda m: dr(e, m) < 0.1]

    eles['near_mcs'] = lambda reco_e: very_near2(mc_part, reco_e)

    eles['hasMC'] = lambda e: e.near_mcs.Count() > 0
    good_eles_with_mc = eles[eles.hasMC]
    good_eles_with_mc['mc'] = lambda e: e.near_mcs.First().ptgev

    make_local(good_eles_with_mc.mc)

    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select('lambda e0001: (e0001.Electrons("Electrons"), e0001)')
        .Select('lambda e0002: (e0002[0].Where(lambda e0032: (e0002[1]'
                '.TruthParticles("TruthParticles")'
                '.Where(lambda e0012: (DeltaR(e0032.eta()) < 0.1)).Count() > 0)), e0002[1])')
        .Select('lambda e0017: e0017[0].Select(lambda e0033: '
                'e0017[1].TruthParticles("TruthParticles")'
                '.Where(lambda e0025: (DeltaR(e0033.eta()) < 0.1)))')
        .Select('lambda e0026: e0026.Select(lambda e0034: e0034.First())')
        .Select('lambda e0027: e0027.Select(lambda e0035: e0035.ptgev())')
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt
