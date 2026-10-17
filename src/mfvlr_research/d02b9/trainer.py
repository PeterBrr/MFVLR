"""Preparation interface; no optimizer or training execution implemented."""
from mfvlr_research.d02b8.contracts import config as validate_base
from .data import loader

def validate_config(config):
    validate_base(config)
    return dict(execution='DISABLED', optimizer='DISABLED', numerical_acceptance='BLOCKED')

class PreparedTrainer:
    def __init__(self, config):
        validate_config(config)
        self.config = dict(config)
    def batches(self, dataset):
        if dataset.manifest_sha256 != self.config['dataset_manifest_sha256']:
            raise ValueError('Dataset/config provenance mismatch')
        return loader(dataset, batch_size=self.config['batch'])
    def fit(self, *args, **kwargs): raise PermissionError('Training requires separate approval')
    def step(self, *args, **kwargs): raise PermissionError('Optimizer steps disabled')
    def backward(self, *args, **kwargs): raise PermissionError('Full-model backward disabled')
