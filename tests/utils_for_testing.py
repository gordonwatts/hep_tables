import ast
from json import dumps, loads
import logging
import os
import re
import shutil
import tempfile
from unittest import mock

from func_adl import EventDataset
import pytest
import servicex.servicex as fe

import hep_tables.local as hep_local
from hep_tables.utils import reset_new_var_counter


# For use in testing - a mock.
f = EventDataset('locads://bogus')

# dump out logs
logging.basicConfig(level=logging.NOTSET)


@pytest.fixture(autouse=True)
def delete_default_downloaded_files():
    download_location = os.path.join(tempfile.gettempdir(), 'servicex-testing')
    import servicex.utils as sx
    sx.default_file_cache_name = download_location
    if os.path.exists(download_location):
        shutil.rmtree(download_location)
    yield
    if os.path.exists(download_location):
        shutil.rmtree(download_location)


@pytest.fixture(autouse=True)
def reset_var_counter():
    # Always start from zero
    reset_new_var_counter()
    # This is the col name in our dummy data
    hep_local.default_col_name = b'JetPt'


@pytest.fixture(scope="module")
def reduce_wait_time():
    old_value = fe.servicex_status_poll_time
    fe.servicex_status_poll_time = 0.01
    yield None
    fe.servicex_status_poll_time = old_value


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

    linq = expr.value(translate)

    # Replace all the eX's in order so that
    # we don't have to keep re-writing when the algorithm changes.

    return clean_linq(linq)


def clean_linq(linq: str) -> str:
    '''
    Noramlize the variables in a linq expression. Should make the
    linq expression more easily comparable even if the algorithm that
    generates the underlying variable numbers changes.
    '''
    all_uses = re.findall('e[0-9]+', linq)
    index = 0
    used = []
    mapping = {}
    for v in all_uses:
        if v not in used:
            used.append(v)
            new_var = f'a{index}'
            index += 1
            mapping[v] = new_var

    max_len = max([len(k) for k in mapping.keys()])
    for l in range(max_len, 0, -1):
        for k in mapping.keys():
            if len(k) == l:
                linq = linq.replace(k, mapping[k])

    return linq
