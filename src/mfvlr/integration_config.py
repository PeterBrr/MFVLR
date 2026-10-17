"""Closed approved integration schema, accepted schemas remain unchanged."""
import copy
import json
from pathlib import Path
from .config import ConfigError,_read

REFERENCE=json.loads(Path(__file__).with_name('integration_contract_snapshot.json').read_text())


def exact(actual,expected,path):
    if type(actual) is not type(expected):raise ConfigError(path+': wrong type')
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise ConfigError(path+': unknown/missing keys')
        for key,value in expected.items():exact(actual[key],value,path+'.'+key)
    elif isinstance(expected,list):
        if len(actual)!=len(expected):raise ConfigError(path+': length')
        for i,value in enumerate(expected):exact(actual[i],value,path+f'[{i}]')
    elif actual!=expected:raise ConfigError(path+': unapproved value')


def load_integration_config(path):
    data=_read(Path(path))
    expected=copy.deepcopy(REFERENCE)
    if data.get('profile')=='debug':expected['profile']='debug'
    exact(data,expected,'integration')
    return copy.deepcopy(data)
