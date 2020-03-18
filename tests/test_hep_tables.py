from hep_tables import xaod_table, make_local
from func_adl import EventDataset
import logging

# For use in testing - a mock.
f = EventDataset('locads://bogus')

# dump out logs
logging.basicConfig(level=logging.NOTSET)


def test_create_base():
    _ = xaod_table(f)


def test_collect_pts():
    df = xaod_table(f)
    seq = df.jets.pt
    a = make_local(seq)
    assert a is not None
