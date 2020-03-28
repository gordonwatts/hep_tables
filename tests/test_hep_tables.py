from hep_tables import make_local, xaod_table
from .utils_for_testing import f, reduce_wait_time, reset_var_counter # NOQA
from .utils_for_testing import files_back_1, good_transform_request # NOQA
from .utils_for_testing import translate_linq


def test_create_base():
    _ = xaod_table(f)


def test_collect_pts(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.pt
    a = make_local(seq)
    assert a is not None
    assert len(a) == 283458
    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_abs_of_data(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = abs(df.jets.pt)
    a = make_local(seq)
    assert a is not None
    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
                         .Select("lambda e5: e5.Select(lambda e3: abs(e3))")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_abs_of_top_leveldata(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = abs(df.met)
    a = make_local(seq)
    assert a is not None
    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.met()")
                         .Select("lambda e2: abs(e2)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_collect_xaod_jet_pts(good_transform_request, reduce_wait_time, files_back_1):
    'Do this with the actual call we need in ATLAS'
    df = xaod_table(f)
    seq = df.Jets("AntiKT4").pt
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.Jets('AntiKT4')")
                         .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_collect_xaod_ele_pts(good_transform_request, reduce_wait_time, files_back_1):
    'Do this with the actual call we need in ATLAS'
    df = xaod_table(f)
    seq = df.Electrons("Electrons").pt
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.Electrons('Electrons')")
                         .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_collect_xaod_call_with_number(good_transform_request, reduce_wait_time, files_back_1):
    'Do this with the actual call we need in ATLAS'
    df = xaod_table(f)
    seq = df.Jets(22.0).pt
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.Jets(22.0)")
                         .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_pt_div(good_transform_request, reduce_wait_time, files_back_1):
    'Do this with the actual call we need in ATLAS'
    df = xaod_table(f)
    seq = df.jets.pt / 1000.0
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
                         .Select("lambda e5: e5.Select(lambda e3: e3/1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_pt_mult(good_transform_request, reduce_wait_time, files_back_1):
    'Do this with the actual call we need in ATLAS'
    df = xaod_table(f)
    seq = df.jets.pt * 1000.0
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
                         .Select("lambda e5: e5.Select(lambda e3: e3 * 1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_pt_add(good_transform_request, reduce_wait_time, files_back_1):
    'Do this with the actual call we need in ATLAS'
    df = xaod_table(f)
    seq = df.jets.pt + 1000.0
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
                         .Select("lambda e5: e5.Select(lambda e3: e3 + 1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_pt_sub(good_transform_request, reduce_wait_time, files_back_1):
    'Do this with the actual call we need in ATLAS'
    df = xaod_table(f)
    seq = df.jets.pt - 1000.0
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
                         .Select("lambda e5: e5.Select(lambda e3: e3 - 1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_jet_pt_filter_pts_gt(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.pt[df.jets.pt > 30.0]
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e5: e5.Select(lambda e2: e2.pt())")
                         .Select("lambda e6: e6.Where(lambda e4: e4 > 30.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_filter_and_divide(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.pt[df.jets.pt > 30.0] / 1000.0
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e6: e6.Select(lambda e2: e2.pt())")
                         .Select("lambda e7: e7.Where(lambda e4: e4 > 30.0)")
                         .Select("lambda e8: e8.Select(lambda e5: e5 / 1000.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_jet_pt_filter_pts_ge(good_transform_request, reduce_wait_time, files_back_1):
    'Do this with the actual call we need in ATLAS'
    df = xaod_table(f)
    seq = df.jets.pt[df.jets.pt >= 30.0]
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e5: e5.Select(lambda e2: e2.pt())")
        .Select("lambda e6: e6.Where(lambda e4: e4 >= 30.0)")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_jet_pt_filter_pts_lt(good_transform_request, reduce_wait_time, files_back_1):
    'Do this with the actual call we need in ATLAS'
    df = xaod_table(f)
    seq = df.jets.pt[df.jets.pt < 30.0]
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e5: e5.Select(lambda e2: e2.pt())")
                         .Select("lambda e6: e6.Where(lambda e4: e4 < 30.0)")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_jet_pt_filter_pts_le(good_transform_request, reduce_wait_time, files_back_1):
    'Do this with the actual call we need in ATLAS'
    df = xaod_table(f)
    seq = df.jets.pt[df.jets.pt <= 30.0]
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e5: e5.Select(lambda e2: e2.pt())")
        .Select("lambda e6: e6.Where(lambda e4: e4 <= 30.0)")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_jet_pt_filter_pts_eq(good_transform_request, reduce_wait_time, files_back_1):
    'Do this with the actual call we need in ATLAS'
    df = xaod_table(f)
    seq = df.jets.pt[df.jets.pt == 30.0]
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e5: e5.Select(lambda e2: e2.pt())")
        .Select("lambda e6: e6.Where(lambda e4: e4 == 30.0)")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_jet_pt_filter_pts_ne(good_transform_request, reduce_wait_time, files_back_1):
    'Do this with the actual call we need in ATLAS'
    df = xaod_table(f)
    seq = df.jets.pt[df.jets.pt != 30.0]
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e5: e5.Select(lambda e2: e2.pt())")
        .Select("lambda e6: e6.Where(lambda e4: e4 != 30.0)")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_filter_jet_objects(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets[df.jets.pt > 30].pt
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e6: e6.Where(lambda e4: e4.pt() > 30)")
        .Select("lambda e7: e7.Select(lambda e5: e5.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_filter_and(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets[(df.jets.pt > 30.0) & (df.jets.pt > 40.0)].pt
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e9: e9.Where(lambda e7: (e7.pt() > 30.0) and (e7.pt() > 40.0))")
        .Select("lambda e10: e10.Select(lambda e8: e8.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_filter_and_abs(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets[(df.jets.pt > 30.0) & (abs(df.jets.eta) < 2.5)].pt
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e10: e10.Where(lambda e8: (e8.pt() > 30.0) and (abs(e8.eta()) < 2.5))")
        .Select("lambda e11: e11.Select(lambda e9: e9.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_binop_in_filter(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets[(df.jets.pt / 1000.0) > 30].pt
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e7: e7.Where(lambda e5: e5.pt()/1000.0 > 30)")
        .Select("lambda e8: e8.Select(lambda e6: e6.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_count_of_objects(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.Count()
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e2: e2.Count()")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_count_of_values(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.pt.Count()
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
        .Select("lambda e3: e3.Count()")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt


def test_count_at_eventLevel(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df[df.jets.Count() == 2].jets.pt
    make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Where("lambda e4: e4.jets().Count() == 2")
        .Select("lambda e5: e5.jets()")
        .Select("lambda e7: e7.Select(lambda e6: e6.pt())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt

# def test_count_in_nested_filter(good_transform_request, reduce_wait_time, files_back_1):
#     df = xaod_table(f)
#     seq1 = df.jets[df.jets.pt > 20000.0]
#     seq2 = seq1.jets[seq1.Count() == 2].pt
#     make_local(seq2)
#     json = good_transform_request
#     txt = translate_linq(
#         f
#         .Select("lambda e1: e1.jets()")
#         .Where("lambda e8: e8.Select(lambda e9: e9.pt() > 20000.0)")
#         .Where("lambda e8: e8.Count() == 2")
#         .Select("lambda e2: e2.Select(lambda e3: e3.pt())")
#         .AsROOTTTree("file.root", "treeme", ['col1']))
#     assert json['selection'] == txt


# def test_math_func_in_filter(good_transform_request, reduce_wait_time, files_back_1):
#     df = xaod_table(f)
#     seq = df.jets[abs(df.jets.eta) < 2.5].pt
#     make_local(seq)
#     json = good_transform_request
#     txt = translate_linq(
#         f
#         .Select("lambda e1: e1.jets()")
#         .Select("lambda e7: e7.Where(lambda e5: abs(e5.eta()) < 2.5)")
#         .Select("lambda e8: e8.Select(lambda e6: e6.pt())")
#         .AsROOTTTree("file.root", "treeme", ['col1']))
#     assert json['selection'] == txt
# def test_filter_on_single_object():
#     df = xaod_table(f)
#     seq = df[df.met > 30.0].jets.pt
#     # make_local(seq)
#     assert False


# def test_count_in_simple_filter(good_transform_request, reduce_wait_time, files_back_1):
#     df = xaod_table(f)
#     seq = df.jets.pt[df.jets.pt.Count() == 2]
#     make_local(seq)
#     json = good_transform_request
#     txt = translate_linq(
#         f
#         .Select("lambda e1: e1.jets()")
#         .Select("lambda e6: e6.Select(lambda e2: e2.pt())")
#         .Where("lambda e5: e5.Count() == 2")
#         .AsROOTTTree("file.root", "treeme", ['col1']))
#     assert json['selection'] == txt


# def test_count_in_called_filter(good_transform_request, reduce_wait_time, files_back_1):
#  Commented out b.c. we are trying to filter at the event level, which is not making sense
#  here. The result is not correct here and is more complex.
#     df = xaod_table(f)
#     seq = df.jets[df.jets.pt.Count() == 2].pt
#     make_local(seq)
#     json = good_transform_request
#     txt = translate_linq(
#         f
#         .Select("lambda e1: e1.jets()")
#         .Where("lambda e5: e5.Select(lambda e6: e6.pt()).Count() == 2")
#         .Select("lambda e8: e8.Select(lambda e7: e7.pt())")
#         .AsROOTTTree("file.root", "treeme", ['col1']))
#     assert json['selection'] == txt
