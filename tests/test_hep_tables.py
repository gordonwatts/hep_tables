import ast
from json import dumps, loads
import logging
import shutil
from unittest import mock

from func_adl import EventDataset
import pytest
import servicex as fe

from hep_tables import make_local, xaod_table

# For use in testing - a mock.
f = EventDataset('locads://bogus')

# dump out logs
logging.basicConfig(level=logging.NOTSET)


@pytest.fixture(autouse=True)
def reset_var_counter():
    import hep_tables
    hep_tables.utils.reset_new_var_counter()


@pytest.fixture(scope="module")
def reduce_wait_time():
    old_value = fe.servicex.servicex_status_poll_time
    fe.servicex.servicex_status_poll_time = 0.01
    yield None
    fe.servicex.servicex_status_poll_time = old_value


def make_minio_file(fname):
    r = mock.MagicMock()
    r.object_name = fname
    return r


class ClientSessionMocker:
    def __init__(self, text, status):
        self._text = text
        self.status = status

    async def text(self):
        return self._text

    async def json(self):
        return loads(self._text)

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self


def good_copy(a, b, c):
    'Mock the fget_object from minio by copying out our test file'
    shutil.copy('tests/sample_servicex_output.root', c)


@pytest.fixture()
def good_transform_request(mocker):
    '''
    Setup to run a good transform request that returns a single file.
    '''
    called_json_data = {}

    def call_post(data_dict_to_save: dict, json=None):
        data_dict_to_save.update(json)
        return ClientSessionMocker(dumps({"request_id": "1234-4433-111-34-22-444"}), 200)
    mocker.patch('aiohttp.ClientSession.post', side_effect=lambda _,
                 json: call_post(called_json_data, json=json))

    r2 = ClientSessionMocker(dumps({"files-remaining": "0", "files-processed": "1"}), 200)
    mocker.patch('aiohttp.ClientSession.get', return_value=r2)

    return called_json_data


@pytest.fixture()
def files_back_1(mocker):
    mocker.patch('minio.api.Minio.list_objects', return_value=[make_minio_file('root:::dcache-atlas-xrootd-wan.desy.de:1094::pnfs:desy.de:atlas:dq2:atlaslocalgroupdisk:rucio:mc15_13TeV:8a:f1:DAOD_STDM3.05630052._000001.pool.root.198fbd841d0a28cb0d9dfa6340c890273-1.part.minio')])  # NOQA
    mocker.patch('minio.api.Minio.fget_object', side_effect=good_copy)
    return None


def translate_linq(expr) -> str:
    '''
    expr is the LINQ expression, short of the value. We return the `qastle` AST.
    '''
    def translate(a: ast.AST):
        import qastle
        return qastle.python_ast_to_text_ast(a)

    return expr.value(translate)


def test_create_base():
    _ = xaod_table(f)


def test_collect_pts(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.pt
    a = make_local(seq)
    assert a is not None
    assert len(a[b'JetPt']) == 283458
    json = good_transform_request
    txt = translate_linq(f
                         .Select("lambda e1: e1.jets()")
                         .Select("lambda e3: e3.Select(lambda e2: e2.pt())")
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
    seq = df.jets.pt/1000.0
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
    seq = df.jets.pt*1000.0
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
    seq = df.jets.pt[df.jets.pt > 30.0]/1000.0
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


# def test_filter_on_single_object():
#     df = xaod_table(f)
#     seq = df[df.met > 30.0].jets.pt
#     # make_local(seq)
#     assert False
