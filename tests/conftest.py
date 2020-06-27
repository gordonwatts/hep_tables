import ast
from json import dumps, loads
import logging
import os
import shutil
import tempfile
from unittest import mock

from servicex import ServiceXDataset
import pytest

import hep_tables.local as hep_local
from hep_tables.utils import reset_new_var_counter
from dataframe_expressions.alias import _reset_alias_catalog
from servicex import clean_linq
import asyncmock


# dump out logs
logging.basicConfig(level=logging.NOTSET)


@pytest.fixture
def servicex_ds(mocker):
    'Create a mock ServiceXDataset'

    # Just pattern it off the real one.
    x = asyncmock.AsyncMock(spec=ServiceXDataset)

    # We need to return a loaded file.
    import uproot
    f_in = uproot.open('tests/sample_servicex_output.root')
    try:
        r = f_in[f_in.keys()[0]]
        data = r.arrays()  # type: ignore
    finally:
        f_in._context.source.close()

    x.get_data_awkward_async.return_value = data

    return x


def extract_selection(ds):
    'Extract the selection from the magic mock for the dataset for the last call'
    return ds.get_data_awkward_async.call_args[0][0]


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

    # And reset the alias
    _reset_alias_catalog()
    yield None

    # For good measure
    reset_new_var_counter()
    _reset_alias_catalog()


@pytest.fixture(scope="module")
def reduce_wait_time():
    import servicex.servicex_adaptor as fe
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

    count = 0

    def call_post(data_dict_to_save: dict, json=None):
        data_dict_to_save.update(json)
        nonlocal count
        count += 1
        return ClientSessionMocker(dumps({"request_id": f"1234-4433-111-34-22-444-{count}"}), 200)

    mocker.patch('aiohttp.ClientSession.post', side_effect=lambda _, json:
                 call_post(called_json_data, json=json))

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
    async def translate(a: ast.AST):
        import qastle
        return qastle.python_ast_to_text_ast(a)

    linq = expr.value(translate)

    # Replace all the eX's in order so that
    # we don't have to keep re-writing when the algorithm changes.

    return clean_linq(linq)
