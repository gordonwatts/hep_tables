from hep_tables import make_local, xaod_table, histogram
from .utils_for_testing import f, reduce_wait_time, reset_var_counter # NOQA
from .utils_for_testing import files_back_1, good_transform_request # NOQA
from .utils_for_testing import translate_linq
from typing import Tuple


def test_numpy_histogram(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = histogram(df.met)
    h = make_local(seq)
    json = good_transform_request
    txt = translate_linq(
        f
        .Select("lambda e1: e1.met()")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert json['selection'] == txt

    assert h is not None
    assert isinstance(h, Tuple)
    assert len(h) == 2

    contents = h[0]
    assert len(contents) == 10
