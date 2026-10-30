"""GenFace/Diffae dataset adapter for the current internal pilot."""
from pathlib import Path

from .pilot_data import PilotFaces


def dataset_for_config(root, config, split, *, tokens=False, limit=None):
    if config.get('dataset') is not None:
        raise ValueError('Unsupported dataset; this pilot supports GenFace real/Diffae only')
    return PilotFaces(Path(root) / config['manifests'][split], tokens=tokens, limit=limit)


def audit_source_groups(partitions):
    """GenFace manifests do not provide a verified subject/source grouping."""
    return {'source_video_disjoint_verified': False}
