from hep_tables import xaod_table, make_local
from func_adl import EventDataset
import logging
import pytest
import servicex as fe
from unittest import mock
import shutil
from json import dumps, loads

# For use in testing - a mock.
f = EventDataset('locads://bogus')

# dump out logs
logging.basicConfig(level=logging.NOTSET)


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


def test_create_base():
    _ = xaod_table(f)


def test_collect_pts(good_transform_request, reduce_wait_time, files_back_1):
    df = xaod_table(f)
    seq = df.jets.pt
    a = make_local(seq)
    assert a is not None
    assert len(a) == 283458
    json = good_transform_request
    assert json['selection'] == "(call ResultTTree (call Select (call SelectMany (call EventDataset (list 'locads://bogus')) (lambda (list e) (call (attr e 'jets')))) (lambda (list e) (call (attr e 'pt')))) (list 'col1') 'treeme' 'file.root')"  # NOQA
