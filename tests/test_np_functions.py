import numpy as np
import pytest
from func_adl_xAOD import ServiceXDatasetSource
from servicex import clean_linq

from hep_tables import make_local, xaod_table

from .conftest import extract_selection, translate_linq


@pytest.mark.parametrize("apply_f, func_text", [
    (lambda f1, f2: np.arctan2(f1, f2), 'atan2'),  # type: ignore
    (lambda f1, f2: np.ldexp(f1, f2), 'ldexp'),  # type: ignore
    (lambda f1, f2: np.power(f1, f2), 'pow'),  # type: ignore
    (lambda f1, f2: np.remainder(f1, f2), 'remainder'),  # type: ignore
    (lambda f1, f2: np.copysign(f1, f2), 'copysign'),  # type: ignore
    (lambda f1, f2: np.nextafter(f1, f2), 'nextafter'),  # type: ignore
    (lambda f1, f2: np.fmod(f1, f2), 'fmod'),  # type: ignore
    (lambda f1, f2: np.fmax(f1, f2), 'fmax'),  # type: ignore
    (lambda f1, f2: np.fmin(f1, f2), 'fmin'),  # type: ignore
])
def test_numpy_functions_2arg(apply_f, func_text, servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = apply_f(df.met, df.met)
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.met()")
        .Select(f"lambda e1: {func_text}(e1, e1)")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


@pytest.mark.parametrize("apply_f, func_text", [
    (lambda f: np.sin(f), 'sin'),  # type: ignore
    (lambda f: np.cos(f), 'cos'),  # type: ignore
    (lambda f: np.tan(f), 'tan'),  # type: ignore
    (lambda f: np.arcsin(f), 'asin'),  # type: ignore
    (lambda f: np.arccos(f), 'acos'),  # type: ignore
    (lambda f: np.arctan(f), 'atan'),  # type: ignore
    (lambda f: np.sinh(f), 'sinh'),  # type: ignore
    (lambda f: np.cosh(f), 'cosh'),  # type: ignore
    (lambda f: np.tanh(f), 'tanh'),  # type: ignore
    (lambda f: np.arcsinh(f), 'asinh'),  # type: ignore
    (lambda f: np.arccosh(f), 'acosh'),  # type: ignore
    (lambda f: np.arctanh(f), 'atanh'),  # type: ignore
    (lambda f: np.exp(f), 'exp'),  # type: ignore
    (lambda f: np.log(f), 'log'),  # type: ignore
    (lambda f: np.log10(f), 'log10'),  # type: ignore
    (lambda f: np.exp2(f), 'exp2'),  # type: ignore
    (lambda f: np.log1p(f), 'log1p'),  # type: ignore
    (lambda f: np.log2(f), 'log2'),  # type: ignore
    (lambda f: np.sqrt(f), 'sqrt'),  # type: ignore
    (lambda f: np.cbrt(f), 'cbrt'),  # type: ignore
    (lambda f: np.ceil(f), 'ceil'),  # type: ignore
    (lambda f: np.floor(f), 'floor'),  # type: ignore
    (lambda f: np.trunc(f), 'trunc'),  # type: ignore
    (lambda f: np.rint(f), 'rint'),  # type: ignore
    (lambda f: np.absolute(f), 'abs'),  # type: ignore
    (lambda f: np.abs(f), 'abs'),
])
def test_numpy_functions(apply_f, func_text, servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = apply_f(df.met)
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.met()")
        .Select(f"lambda e1: {func_text}(e1)")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


def test_numpy_2arg_func(servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)

    df_1 = df.jets.pt
    df_2 = df.jets.eta

    seq = np.arctan2(df_1, df_2)  # type: ignore
    make_local(seq)

    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select("lambda e1: e1.jets()")
        .Select("lambda e1: e1.Select(lambda e2: atan2(e2.pt(), e2.eta()))")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt
