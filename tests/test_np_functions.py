import numpy as np
import pytest
from func_adl_xAOD import ServiceXDatasetSource
from servicex import clean_linq

from hep_tables import make_local, xaod_table

from .conftest import extract_selection, translate_linq


@pytest.mark.parametrize("apply_f, func_text", [
    (lambda f1, f2: np.arctan2(f1, f2), 'atan2'),
    (lambda f1, f2: np.ldexp(f1, f2), 'ldexp'),
    (lambda f1, f2: np.power(f1, f2), 'pow'),
    (lambda f1, f2: np.remainder(f1, f2), 'remainder'),
    (lambda f1, f2: np.copysign(f1, f2), 'copysign'),
    (lambda f1, f2: np.nextafter(f1, f2), 'nextafter'),
    (lambda f1, f2: np.fmod(f1, f2), 'fmod'),
    (lambda f1, f2: np.fmax(f1, f2), 'fmax'),
    (lambda f1, f2: np.fmin(f1, f2), 'fmin'),
])
def test_numpy_functions_2arg(apply_f, func_text, servicex_ds):
    f = ServiceXDatasetSource(servicex_ds)
    df = xaod_table(f)
    seq = apply_f(df.met, df.met)
    make_local(seq)
    selection = extract_selection(servicex_ds)
    txt = translate_linq(
        f
        .Select(f"lambda e1: {func_text}(e1.met(), e1.met())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt


@pytest.mark.parametrize("apply_f, func_text", [
    (lambda f: np.sin(f), 'sin'),
    (lambda f: np.cos(f), 'cos'),
    (lambda f: np.tan(f), 'tan'),
    (lambda f: np.arcsin(f), 'asin'),
    (lambda f: np.arccos(f), 'acos'),
    (lambda f: np.arctan(f), 'atan'),
    (lambda f: np.sinh(f), 'sinh'),
    (lambda f: np.cosh(f), 'cosh'),
    (lambda f: np.tanh(f), 'tanh'),
    (lambda f: np.arcsinh(f), 'asinh'),
    (lambda f: np.arccosh(f), 'acosh'),
    (lambda f: np.arctanh(f), 'atanh'),
    (lambda f: np.exp(f), 'exp'),
    (lambda f: np.log(f), 'log'),
    (lambda f: np.log10(f), 'log10'),
    (lambda f: np.exp2(f), 'exp2'),
    (lambda f: np.log1p(f), 'log1p'),
    (lambda f: np.log2(f), 'log2'),
    (lambda f: np.sqrt(f), 'sqrt'),
    (lambda f: np.cbrt(f), 'cbrt'),
    (lambda f: np.ceil(f), 'ceil'),
    (lambda f: np.floor(f), 'floor'),
    (lambda f: np.trunc(f), 'trunc'),
    (lambda f: np.rint(f), 'rint'),
    (lambda f: np.absolute(f), 'abs'),
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
        .Select(f"lambda e1: {func_text}(e1.met())")
        .AsROOTTTree("file.root", "treeme", ['col1']))
    assert clean_linq(selection) == txt
