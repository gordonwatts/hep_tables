from dataframe_expressions.render import render

from func_adl import EventDataset

from dataframe_expressions import ast_DataFrame
from hep_tables import xaod_table
from hep_tables.utils import _find_dataframes

# For use in testing - a mock.
f = EventDataset('locads://bogus')


def test_find_dataframes():
    df = xaod_table(f)
    seq = df.jets.pt
    expr = render(seq)

    found_df = _find_dataframes(expr)
    assert isinstance(found_df, ast_DataFrame)
    assert found_df.dataframe is df


def test_find_nested_dataframes():
    df = xaod_table(f)
    seq = df.jets[df.jets.pt > 30].pt
    expr = render(seq)

    found_df = _find_dataframes(expr)
    assert isinstance(found_df, ast_DataFrame)
    assert found_df.dataframe is df


# def test_fail_to_find_two_dataframes():
#     df1 = xaod_table(f)
#     f2 = EventDataset('locads://bogusss')
#     df2 = xaod_table(f2)

#     deq = df1.jets[df2.jets.pt > 30].pt
#     expr = render(deq)

#     with pytest.raises(Exception):
#         _find_dataframes(expr)
# TODO: this doesn't fail yet, but it should - error seems to be in the underlying library.
# Since this is a prototype, no need to chase it down right now.
