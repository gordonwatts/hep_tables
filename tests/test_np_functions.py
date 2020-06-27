from hep_tables import make_local, xaod_table, histogram
from func_adl_xAOD import ServiceXDatasetSource

from servicex import clean_linq

from .conftest import (
    translate_linq,
    extract_selection
    )
from typing import Tuple


def test_numpy_abs(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    import numpy as np
    seq = np.abs(df.met)
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.met()")
        .Select("lambda e2: abs(e2)")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_numpy_sqrt(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    import numpy as np
    seq = np.sqrt(df.met)  # type: ignore
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.met()")
        .Select("lambda e2: sqrt(e2)")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_numpy_histogram(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = histogram(df.met)
    h = make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.met()")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt

    assert h is not None
    assert isinstance(h, Tuple)
    assert len(h) == 2

    contents = h[0]
    assert len(contents) == 10
