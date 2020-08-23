import ast
from hep_tables.graph_info import v_info
import logging
import os
import shutil
import tempfile
from json import dumps, loads
from typing import Any, Callable, Union
from unittest import mock

import asyncmock
import pytest
from dataframe_expressions.alias import _reset_alias_catalog
from dataframe_expressions.asts import ast_DataFrame
from func_adl.event_dataset import EventDataset
from func_adl.object_stream import ObjectStream
from servicex import ServiceXDataset, clean_linq

import hep_tables.local as hep_local
from hep_tables.hep_table import xaod_table
from hep_tables.transforms import root_sequence_transform, sequence_predicate_base
from hep_tables.utils import QueryVarTracker

# dump out logs
logging.basicConfig(level=logging.NOTSET)


@pytest.fixture
def mock_qt(mocker):
    count = 999

    def return_count():
        nonlocal count
        count += 1
        return f'e{count}'

    qt = mocker.MagicMock(spec=QueryVarTracker)
    qt.new_var_name.side_effect = return_count
    return qt


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


def mock_vinfo(mocker, level: int = 0, node: ast.AST = None, seq: sequence_predicate_base = None, order: int = 0):
    info = mocker.MagicMock(spec=v_info)
    info.level = level
    info.node = node
    info.sequence = seq
    info.order = order
    return info


def extract_selection(ds) -> str:
    'Extract the selection from the magic mock for the dataset for the last call'
    return ds.get_data_awkward_async.call_args[0][0]


class my_events(EventDataset):
    '''Dummy event source'''
    async def execute_result_async(self, a: ast.AST) -> Any:
        pass


@pytest.fixture
def mock_root_sequence_transform(mocker):
    '''Creates a mock sequence transform'''
    mine = my_events()
    a = ast_DataFrame(xaod_table(mine))

    root_seq = mocker.MagicMock(spec=root_sequence_transform)
    root_seq.sequence.return_value = mine

    return mine, a, root_seq


class MatchAST:
    def __init__(self, true_ast: ast.AST):
        '''Match object for an ast'''
        self._true_ast = true_ast

    def clean(self, a: Union[str, ast.AST]):
        base_string = ast.dump(a) if isinstance(a, ast.AST) else a
        return base_string \
            .replace(', annotation=None', '') \
            .replace(', vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]', '') \
            .replace(', ctx=Load()', '')

    def __eq__(self, other: ast.AST):
        other_ast = self.clean(other)
        true_ast = self.clean(self._true_ast)
        if true_ast != other_ast:
            print(f'true: {true_ast}')
            print(f'test: {other_ast}')


class MatchObjectSequence:
    def __init__(self, a_list: ObjectStream):
        from func_adl.ast.func_adl_ast_utils import change_extension_functions_to_calls
        self._asts = [change_extension_functions_to_calls(a_list._ast)]

    def clean(self, a: Union[str, ast.AST]):
        base_string = ast.dump(a) if isinstance(a, ast.AST) else a
        return base_string \
            .replace(', annotation=None', '') \
            .replace(', vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]', '') \
            .replace(', ctx=Load()', '')

    def __eq__(self, other: Union[str, ObjectStream]):
        other_ast = self.clean(other._ast) if isinstance(other, ObjectStream) else other
        r = any(self.clean(a) == other_ast for a in self._asts)
        if not r:
            print(f'test: {other_ast}')
            for a in self._asts:
                print(f'true: {self.clean(a)}')
        return r


class MatchQastleSequence:
    def __init__(self, linq_expr: Callable[[ObjectStream], ObjectStream]):
        class as_qastle(EventDataset):
            async def execute_result_async(self, a: ast.AST) -> Any:
                from func_adl.ast.func_adl_ast_utils import change_extension_functions_to_calls
                a = change_extension_functions_to_calls(a)
                from qastle import python_ast_to_text_ast
                return clean_linq(python_ast_to_text_ast(a))

            def __repr__(self):
                return "'ServiceXDatasetSource'"

        os = linq_expr(as_qastle())
        self._qastle = os.AsROOTTTree('file.root', 'treeme', ['col1']).value()

    def __eq__(self, other: str):
        other = clean_linq(other)
        if self._qastle != other:
            print(f'true: {self._qastle}')
            print(f'test: {other}')
            return False
        else:
            return True


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
    # This is the col name in our dummy data
    hep_local.default_col_name = b'JetPt'

    # And reset the alias
    _reset_alias_catalog()
    yield None

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
