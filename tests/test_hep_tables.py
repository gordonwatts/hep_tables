from dataframe_expressions import DataFrame
from func_adl_xAOD import ServiceXDatasetSource
import pytest
from servicex import clean_linq

from hep_tables import make_local, xaod_table

from .conftest import extract_selection, translate_linq


@pytest.fixture(autouse=True)
def reset_var_counter_alias():
    from dataframe_expressions.alias import _reset_alias_catalog
    _reset_alias_catalog()
    yield None
    _reset_alias_catalog()


def test_create_base(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    _ = xaod_table(f)


def test_create_multibase(servicex_ds):
    f1 = ServiceXDatasetSource(servicex_ds)
    f2 = ServiceXDatasetSource(servicex_ds)
    _ = xaod_table(f1, f2)


def test_create_bad_empty():
    with pytest.raises(Exception):
        xaod_table()


def test_create_bad_option():
    with pytest.raises(Exception):
        xaod_table("hi there")


def test_create_bad_multi_option():
    with pytest.raises(Exception):
        xaod_table(["hi there", "dude"])


def test_copy_xaod_table_1(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    x1 = xaod_table(f)
    import copy
    x2 = copy.deepcopy(x1)
    assert x1 is x2
    assert isinstance(x1, xaod_table)


def test_copy_xaod_table_2(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    x1 = xaod_table(f).jets.pt
    import copy
    x2 = copy.deepcopy(x1)
    assert x1 is not x2
    assert isinstance(x1, DataFrame)


def test_collect_pts(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt
    a = make_local(seq)
    assert a is not None
    assert len(a) == 283458
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_collect_pts_2_files(servicex_ds):
    f1 = ServiceXDatasetSource(servicex_ds)
    f2 = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f1, f2)
    seq = df.jets.pt
    a = make_local(seq)
    assert a is not None
    assert len(a) == 2 * 283458
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f1
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_collect_pts_as_call(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets().pt()
    a = make_local(seq)
    assert a is not None
    assert len(a) == 283458
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_abs_of_data(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = abs(df.jets.pt)
    a = make_local(seq)
    assert a is not None
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
                         .Select("lambda e5: e5.Select(lambda e3: abs(e3))")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_abs_of_data_with_calls(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = abs(df.jets().pt())
    a = make_local(seq)
    assert a is not None
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
                         .Select("lambda e5: e5.Select(lambda e3: abs(e3))")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_abs_of_top_leveldata(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = abs(df.met)
    a = make_local(seq)
    assert a is not None
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.met()")
                         .Select("lambda e2: abs(e2)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_collect_xaod_jet_pts(servicex_ds):
    'Do this with the actual call we need in ATLAS'
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.Jets("AntiKT4").pt
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.Jets('AntiKT4')")
                         .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_collect_xaod_ele_pts(servicex_ds):
    'Do this with the actual call we need in ATLAS'
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.Electrons("Electrons").pt
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.Electrons('Electrons')")
                         .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_collect_xaod_call_with_number(servicex_ds):
    'Do this with the actual call we need in ATLAS'
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.Jets(22.0).pt
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.Jets(22.0)")
                         .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_pt_div(servicex_ds):
    'Do this with the actual call we need in ATLAS'
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt / 1000.0
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
                         .Select("lambda e5: e5.Select(lambda e3: e3/1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_pt_mult(servicex_ds):
    'Do this with the actual call we need in ATLAS'
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt * 1000.0
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
                         .Select("lambda e5: e5.Select(lambda e3: e3 * 1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_pt_add(servicex_ds):
    'Do this with the actual call we need in ATLAS'
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt + 1000.0
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
                         .Select("lambda e5: e5.Select(lambda e3: e3 + 1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_pt_sub(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt - 1000.0
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
                         .Select("lambda e5: e5.Select(lambda e3: e3 - 1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_pt_good(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt > 1000.0
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
                         .Select("lambda e5: e5.Select(lambda e3: e3 > 1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_jet_pt_filter_pts_gt(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt[df.jets.pt > 30.0]
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e5: e5.Select(lambda e2: e2.pt())")
                         .Select("lambda e6: e6.Where(lambda e3: e3 > 30.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_filter_lambda(servicex_ds):
    def good_jet(pt):
        return pt > 30.0

    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt[good_jet]
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e5: e5.Select(lambda e2: e2.pt())")
                         .Select("lambda e6: e6.Where(lambda e3: e3 > 30.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_filter_chain(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq1 = df.jets[df.jets.pt > 30.0]
    seq = seq1[seq1.eta < 2.4].pt
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e6: e6.Where(lambda e3: e3.pt() > 30.0)")
                         .Select("lambda e7: e7.Where(lambda e4: e4.eta() < 2.4)")
                         .Select("lambda e5: e5.Select(lambda e2: e2.pt())")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_filter_and_divide(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt[df.jets.pt > 30.0] / 1000.0
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e6: e6.Select(lambda e2: e2.pt())")
                         .Select("lambda e7: e7.Where(lambda e3: e3 > 30.0)")
                         .Select("lambda e8: e8.Select(lambda e5: e5 / 1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_filter_and_divide_with_call(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets().pt[df.jets().pt > 30.0] / 1000.0
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e6: e6.Select(lambda e2: e2.pt())")
                         .Select("lambda e7: e7.Where(lambda e3: e3 > 30.0)")
                         .Select("lambda e8: e8.Select(lambda e5: e5 / 1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_jet_pt_filter_pts_ge(servicex_ds):
    'Do this with the actual call we need in ATLAS'
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt[df.jets.pt >= 30.0]
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e5: e5.Select(lambda e2: e2.pt())")
        .Select("lambda e6: e6.Where(lambda e4: e4 >= 30.0)")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_jet_pt_filter_pts_lt(servicex_ds):
    'Do this with the actual call we need in ATLAS'
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt[df.jets.pt < 30.0]
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e5: e5.Select(lambda e2: e2.pt())")
                         .Select("lambda e6: e6.Where(lambda e4: e4 < 30.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_jet_pt_filter_pts_le(servicex_ds):
    'Do this with the actual call we need in ATLAS'
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt[df.jets.pt <= 30.0]
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e5: e5.Select(lambda e2: e2.pt())")
        .Select("lambda e6: e6.Where(lambda e4: e4 <= 30.0)")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_jet_pt_filter_pts_eq(servicex_ds):
    'Do this with the actual call we need in ATLAS'
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt[df.jets.pt == 30.0]
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e5: e5.Select(lambda e2: e2.pt())")
        .Select("lambda e6: e6.Where(lambda e4: e4 == 30.0)")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_jet_pt_filter_pts_ne(servicex_ds):
    'Do this with the actual call we need in ATLAS'
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt[df.jets.pt != 30.0]
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e5: e5.Select(lambda e2: e2.pt())")
        .Select("lambda e6: e6.Where(lambda e4: e4 != 30.0)")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_filter_jet_objects(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets[df.jets.pt > 30].pt
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e7: e7.Where(lambda e2: e2.pt() > 30)")
        .Select("lambda e8: e8.Select(lambda e6: e6.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_filter_jet_by_attribute(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets[df.jets.hasProdVtx].pt
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e7: e7.Where(lambda e2: e2.hasProdVtx())")
        .Select("lambda e8: e8.Select(lambda e6: e6.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_filter_jet_by_attributes(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets[df.jets.hasProdVtx & df.jets.hasDecayVtx].pt
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e7: e7.Where(lambda e2: e2.hasProdVtx() and e2.hasDecayVtx())")
        .Select("lambda e8: e8.Select(lambda e6: e6.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_filter_and(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets[(df.jets.pt > 30.0) & (df.jets.pt > 40.0)].pt
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e9: e9.Where(lambda e7: (e7.pt() > 30.0) and (e7.pt() > 40.0))")
        .Select("lambda e10: e10.Select(lambda e8: e8.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_filter_or(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets[(df.jets.pt > 30.0) | (df.jets.pt > 40.0)].pt
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e9: e9.Where(lambda e7: (e7.pt() > 30.0) or (e7.pt() > 40.0))")
        .Select("lambda e10: e10.Select(lambda e8: e8.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_filter_not(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets[~(df.jets.pt > 30.0)].pt
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e9: e9.Where(lambda e7: not (e7.pt() > 30.0))")
        .Select("lambda e10: e10.Select(lambda e8: e8.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_filter_and_abs(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets[(df.jets.pt > 30.0) & (abs(df.jets.eta) < 2.5)].pt
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e10: e10.Where(lambda e8: (e8.pt() > 30.0) and (abs(e8.eta()) < 2.5))")
        .Select("lambda e11: e11.Select(lambda e9: e9.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_binop_in_filter(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets[(df.jets.pt / 1000.0) > 30].pt
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e7: e7.Where(lambda e5: e5.pt()/1000.0 > 30)")
        .Select("lambda e8: e8.Select(lambda e6: e6.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_count_of_events(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.Count()
    with pytest.raises(Exception) as e:
        make_local(seq)

    assert 'Count' in str(e.value)


def test_count_of_objects(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.Count()
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e2: e2.Count()")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_count_of_values(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt.Count()
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
        .Select("lambda e3: e3.Count()")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_count_at_eventLevel(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df[df.jets.Count() == 2].jets.pt
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Where("lambda e4: e4.jets().Count() == 2")
        .Select("lambda e5: e5.jets()")
        .Select("lambda e7: e7.Select(lambda e6: e6.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_first_at_object_level(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.First().pt
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e5: e5.jets()")
        .Select("lambda e7: e7.First()")
        .Select("lambda e8: e8.pt()")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_first_at_leaf_level(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt.First()
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e5: e5.jets()")
        .Select("lambda e7: e7.Select(lambda e4: e4.pt())")
        .Select("lambda e9: e9.First()")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_make_local_twice(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt
    make_local(seq)
    json_1 = clean_linq(extract_selection(servicex_ds))

    make_local(seq)
    json_2 = clean_linq(extract_selection(servicex_ds))

    assert json_1 == json_2


def test_make_local_twice_check_test(servicex_ds):
    # Make sure this method of testing continues to work
    # references and dicts in python are funny!
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt
    make_local(seq)
    json_1 = clean_linq(extract_selection(servicex_ds))

    make_local(seq / 1000.0)
    json_2 = clean_linq(extract_selection(servicex_ds))

    assert json_1 != json_2


def test_make_local_twice_filter(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets[df.jets.pt > 30].pt
    make_local(seq)
    json_1 = clean_linq(extract_selection(servicex_ds))

    make_local(seq)
    json_2 = clean_linq(extract_selection(servicex_ds))

    assert json_1 == json_2


# def test_count_in_nested_filter(servicex_ds):
#     df = xaod_table(f)
#     seq1 = df.jets[df.jets.pt > 20000.0]
#     seq2 = seq1.jets[seq1.Count() == 2].pt
#     make_local(seq2)
#     selection = extract_selection(servicex_ds)
#     txt = translate_linq(
#         f
#         .Select("lambda e1: e1.jets()")
#         .Where("lambda e8: e8.Select(lambda e9: e9.pt() > 20000.0)")
#         .Where("lambda e8: e8.Count() == 2")
#         .Select("lambda e2: e2.Select(lambda e3: e3.pt())")
#         .AsROOTTTree("file.root", "treeme", ['col1']))
#     assert clean_linq(selection) == txt


# def test_math_func_in_filter(servicex_ds):
#     df = xaod_table(f)
#     seq = df.jets[abs(df.jets.eta) < 2.5].pt
#     make_local(seq)
#     selection = extract_selection(servicex_ds)
#     txt = translate_linq(
#         f
#         .Select("lambda e1: e1.jets()")
#         .Select("lambda e7: e7.Where(lambda e5: abs(e5.eta()) < 2.5)")
#         .Select("lambda e8: e8.Select(lambda e6: e6.pt())")
#         .AsROOTTTree("file.root", "treeme", ['col1']))
#     assert clean_linq(selection) == txt
# def test_filter_on_single_object():
#     df = xaod_table(f)
#     seq = df[df.met > 30.0].jets.pt
#     # make_local(seq)
#     assert False


# def test_count_in_simple_filter(servicex_ds):
#     df = xaod_table(f)
#     seq = df.jets.pt[df.jets.pt.Count() == 2]
#     make_local(seq)
#     selection = extract_selection(servicex_ds)
#     txt = translate_linq(
#         f
#         .Select("lambda e1: e1.jets()")
#         .Select("lambda e6: e6.Select(lambda e2: e2.pt())")
#         .Where("lambda e5: e5.Count() == 2")
#         .AsROOTTTree("file.root", "treeme", ['col1']))
#     assert clean_linq(selection) == txt


# def test_count_in_called_filter(servicex_ds):
#  Commented out b.c. we are trying to filter at the event level, which is not making sense
#  here. The result is not correct here and is more complex.
#     df = xaod_table(f)
#     seq = df.jets[df.jets.pt.Count() == 2].pt
#     make_local(seq)
#     selection = extract_selection(servicex_ds)
#     txt = translate_linq(
#         f
#         .Select("lambda e1: e1.jets()")
#         .Where("lambda e5: e5.Select(lambda e6: e6.pt()).Count() == 2")
#         .Select("lambda e8: e8.Select(lambda e7: e7.pt())")
#         .AsROOTTTree("file.root", "treeme", ['col1']))
#     assert clean_linq(selection) == txt
