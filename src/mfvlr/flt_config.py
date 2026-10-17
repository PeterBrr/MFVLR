"""Separate closed full-FLT schema/approval, accepted configs unchanged."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path

from .config import ConfigError, _read
from .foundation_config import _exact
from .tokenizer import verify_assets

_REFERENCE=json.loads(Path(__file__).with_name("flt_contract_snapshot.json").read_text(encoding="utf-8"))


def _validate(data):
    if type(data) is not dict or set(data)!=set(_REFERENCE):
        raise ConfigError("full FLT: unknown/missing fields")
    for key,expected in _REFERENCE.items():
        if key not in ('profile','diagnostics'):
            _exact(data[key],expected,f'flt.{key}')
    if data['profile'] not in ('reference','debug'):
        raise ConfigError('full FLT profile must be reference/debug')
    diagnostics={'batch_size':1,'iterations':0,'enabled':False} if data['profile']=='reference' else {
        'batch_size':2,'iterations':3,'enabled':True}
    _exact(data['diagnostics'],diagnostics,'flt.diagnostics')


@dataclass(frozen=True)
class FLTConfig:
    path: Path
    _data: dict

    @property
    def data(self): return copy.deepcopy(self._data)

    def require_approval(self):
        _validate(self._data)
        verify_assets()


def load_flt_config(path: str|Path) -> FLTConfig:
    path=Path(path).resolve()
    raw=_read(path)
    if 'base' in raw:
        if set(raw)!= {'schema_version','phase','profile','base','merge_policy','approval','overrides'}:
            raise ConfigError('full FLT overlay has unknown/missing fields')
        for key,expected in [('schema_version',1),('phase','2C.2'),('profile','debug'),
                             ('base','flt_reference.yaml'),('merge_policy','diagnostics_only'),
                             ('approval',_REFERENCE['approval'])]:
            _exact(raw[key],expected,f'flt.overlay.{key}')
        if type(raw['overrides']) is not dict or set(raw['overrides'])!={'diagnostics'}:
            raise ConfigError('full FLT overlays may only replace diagnostics')
        base_path=path.parent/raw['base']
        if base_path.resolve()==path: raise ConfigError('full FLT overlay cycle')
        base=_read(base_path)
        if 'base' in base: raise ConfigError('full FLT supports one overlay level only')
        _validate(base)
        if base['profile']!='reference': raise ConfigError('full FLT base must be reference')
        raw=copy.deepcopy(base)
        raw['profile']='debug'
        raw['diagnostics']=copy.deepcopy(_read(path)['overrides']['diagnostics'])
    _validate(raw)
    verify_assets()
    return FLTConfig(path,copy.deepcopy(raw))
