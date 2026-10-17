"""Exclusive metadata ownership and mock recovery; no process launch API."""
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
from .schema import InvalidRecord, canonical, file_hash, fingerprint, write_new

GIB = 1024 ** 3

@dataclass(frozen=True)
class Reservations:
    b1: int = 5 * GIB
    b2: int = 6 * GIB
    disk: int = 8 * GIB

    def __post_init__(self):
        if any(type(v) is not int or v < floor for v, floor in
               ((self.b1, 5*GIB), (self.b2, 6*GIB), (self.disk, 8*GIB))):
            raise InvalidRecord('Reservations may increase, never lower the floors')

    def check(self, batch, available_ram, free_disk):
        if type(batch) is not int or batch not in (1, 2):
            raise InvalidRecord('Batch')
        if any(type(v) is not int or v < 0 for v in (available_ram, free_disk)):
            raise InvalidRecord('Missing/untrusted resource reading')
        ok = available_ram >= (self.b1 if batch == 1 else self.b2) and free_disk >= self.disk
        return {'resource': 'PASS' if ok else 'INSUFFICIENT', 'execution': 'NOT RUN',
                'available_ram': available_ram, 'free_disk': free_disk}

def process_identity(pid, created):
    if type(pid) is not int or pid <= 0 or type(created) not in (float, int) or not math.isfinite(created) or created <= 0:
        raise InvalidRecord('PID + process creation time required')
    return {'pid': pid, 'created': created}

class Coordinator:
    """One permanent claim per root. No automatic stale lock deletion/reuse.

    Liveness provider returns the current creation time or None; errors fail closed.
    A completed lock also cannot be reused. Recovery uses a new attempt path.
    """
    def __init__(self, root, owner, liveness):
        self.root = Path(root)
        self.owner = process_identity(**owner)
        self.liveness = liveness

    def claim(self):
        if self.liveness(self.owner['pid']) != self.owner['created']:
            raise InvalidRecord('Coordinator identity is not live')
        write_new(self.root / 'coordinator.lock.json', self.owner)

    def reserve(self, cell_id, identity_hash):
        from .schema import schedule, sha
        import json
        if cell_id not in {c['experiment_id'] for c in schedule()}:
            raise InvalidRecord('Unregistered cell')
        if json.loads((self.root / 'coordinator.lock.json').read_text()) != self.owner:
            raise InvalidRecord('Coordinator owner changed')
        if self.liveness(self.owner['pid']) != self.owner['created']:
            raise InvalidRecord('PID reused/dead coordinator')
        sha(identity_hash)
        path = self.root / 'cells' / cell_id
        path.mkdir(parents=True, exist_ok=False)
        write_new(path / 'reservation.json', {'identity': identity_hash, 'owner': self.owner, 'execution': 'NOT RUN'})
        return path

def archive_attempt(source, destination, *, owner, liveness):
    """Read-only byte copy, never discard source; refuse live/unknown/reused PID."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    process_identity(**owner)
    observed = liveness(owner['pid'])
    if observed is not None:
        raise InvalidRecord('Owner alive or PID reused: explicit diagnosis required')
    if not source.is_dir() or destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise InvalidRecord('Archive must be a disjoint new directory')
    if any(p.is_symlink() or (hasattr(p, 'is_junction') and p.is_junction()) for p in source.rglob('*')):
        raise InvalidRecord('No linked evidence paths')
    files = {p.relative_to(source).as_posix(): file_hash(p) for p in source.rglob('*') if p.is_file()}
    destination.mkdir(parents=True, exist_ok=False)
    for rel, digest in files.items():
        dest = destination / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with (source / rel).open('rb') as src, dest.open('xb') as out:
            shutil.copyfileobj(src, out)
        if file_hash(dest) != digest or file_hash(source / rel) != digest:
            raise InvalidRecord('Evidence changed during archive')
    if {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()} != set(files):
        raise InvalidRecord('Attempt mutated during archival')
    manifest = {'files': files, 'owner': owner, 'cause': 'UNKNOWN unless separately evidenced', 'source_preserved': True}
    write_new(destination / 'archive_manifest.json', manifest)
    return manifest

def scientific_prefix(original, replay):
    """Compare all complete records, including failures; resource timing excluded explicitly."""
    from .schema import exact_keys
    for records in (original, replay):
        seen = set()
        for row in records:
            exact_keys(row, ('comparison_id', 'scientific', 'resources'))
            exact_keys(row['resources'], ('seconds', 'rss_bytes'))
            if type(row['comparison_id']) is not str or not row['comparison_id'] or row['comparison_id'] in seen:
                raise InvalidRecord('Missing/duplicate comparison identifier')
            seen.add(row['comparison_id'])
            canonical(row)  # Reject NaN/Infinity, not scientific values selected by a whitelist.
    mismatches = [i for i, row in enumerate(original) if i >= len(replay) or
                  canonical([row['comparison_id'], row['scientific']]) !=
                  canonical([replay[i]['comparison_id'], replay[i]['scientific']])]
    return {'status': 'FAIL' if mismatches else 'PASS', 'original_complete_records': len(original),
            'mismatch_indices': mismatches, 'original_sha256': fingerprint(original),
            'replay_sha256': fingerprint(replay), 'ignored_fields': ['resources.seconds', 'resources.rss_bytes']}

def read_complete_prefix(path):
    """Only an unterminated final JSONL fragment may be incomplete; preserve its hash."""
    import hashlib
    import json
    records, fragment = [], b''
    with Path(path).open('rb') as f:
        for line in f:
            if not line.endswith(b'\n'):
                fragment = line
                break
            try:
                records.append(json.loads(line))
            except (ValueError, UnicodeDecodeError) as exc:
                raise InvalidRecord('Malformed complete scientific record') from exc
    return records, {'incomplete_fragment_bytes':len(fragment), 'incomplete_fragment_sha256':hashlib.sha256(fragment).hexdigest()}

def verify_replay(original_path, replay_path, output_path):
    original, fragment = read_complete_prefix(original_path)
    replay, replay_fragment = read_complete_prefix(replay_path)
    report = scientific_prefix(original, replay)
    report.update(original_fragment=fragment, replay_fragment=replay_fragment)
    write_new(output_path, report)  # Preserve mismatch evidence BEFORE hard stopping.
    if report['status'] != 'PASS' or replay_fragment['incomplete_fragment_bytes']:
        raise InvalidRecord('Replay divergence or incomplete replay: dependent work blocked')
    return report

