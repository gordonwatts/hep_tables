from dataframe_expressions import user_func
import pytest

from hep_tables import make_local, xaod_table

from func_adl_xAOD import ServiceXDatasetSource

from servicex import clean_linq

from .conftest import (
    translate_linq,
    extract_selection
    )


def test_user_function_with_implied(servicex_ds):

    @user_func
    def tns(e1: float) -> float:
        assert False, 'this is a fake function and should never be called'

    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    with pytest.raises(Exception):
        seq = tns(df.jets.pt)
        make_local(seq)


def test_user_function_with_map_lambda(servicex_ds):
    @user_func
    def tns(e1: float) -> float:
        assert False, 'this is a fake function and should never be called'

    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt.map(lambda j: tns(j))
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
                         .Select("lambda e5: e5.Select(lambda e3: tns(e3))")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_user_function_with_map_lambda_no_type(servicex_ds):
    @user_func
    def tns(e1):
        assert False, 'this is a fake function and should never be called'

    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt.map(lambda j: tns(j))
    with pytest.raises(Exception) as e:
        make_local(seq)

    assert 'hint' in str(e.value)


def test_user_function_with_map_func(servicex_ds):
    @user_func
    def tns(e1: float) -> float:
        assert False, 'this is a fake function and should never be called'

    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.pt.map(tns)
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e4: e4.Select(lambda e2: e2.pt())")
                         .Select("lambda e5: e5.Select(lambda e3: tns(e3))")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_user_function_with_map_fcall(servicex_ds):
    @user_func
    def tns(e1: float) -> float:
        assert False, 'this is a fake function and should never be called'

    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.map(lambda j: tns(j.pt))
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e5: e5.Select(lambda e2: tns(e2.pt()))")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_user_function_with_map_2fcall(servicex_ds):
    @user_func
    def tns(e1: float, e2: float) -> float:
        assert False, 'this is a fake function and should never be called'

    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.map(lambda j: tns(j.pt, j.eta))
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e7: e7.Select(lambda e2: tns(e2.pt(), e2.eta()))")
                         .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_user_func_with_two_maps(servicex_ds):

    @user_func
    def DeltaR(e1: float, e2: float) -> float:
        assert False, 'this is a fake function and should never be called'

    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = df.jets.map(lambda j: df.Electrons.map(lambda e: DeltaR(e.eta, j.eta)))
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: (e1.jets(), e1)")
        .Select("lambda e14: e14[0].Select(lambda e3: "
                "e14[1]"
                ".Electrons()"
                ".Select(lambda e13: DeltaR(e13.eta(), e3.eta())))")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt
