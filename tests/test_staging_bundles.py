import io
import tarfile

import pytest

from scripts.build_staging_bundles import archive, verify


def test_archive_is_repeatable_and_content_checked(tmp_path):
    first, second = tmp_path / 'first.tar', tmp_path / 'second.tar'
    archive(first, {'payload/incident-block.json': b'{}'}, 'tank')
    archive(second, {'payload/incident-block.json': b'{}'}, 'tank')
    assert first.read_bytes() == second.read_bytes()
    assert not verify(first)['physical_acceptance']
    # A byte change without changing the manifest must fail verification.
    first.write_bytes(first.read_bytes().replace(b'{}', b'[]'))
    with pytest.raises(ValueError, match='manifest'):
        verify(first)


def test_verifier_rejects_paths_and_links_without_extraction(tmp_path):
    path = tmp_path / 'bad.tar'
    for name, kind in [('../outside', tarfile.REGTYPE), ('linked', tarfile.SYMTYPE)]:
        with tarfile.open(path, 'w') as output:
            entry = tarfile.TarInfo(name)
            entry.type = kind
            entry.linkname = '/etc/passwd' if kind == tarfile.SYMTYPE else ''
            output.addfile(entry, io.BytesIO(b''))
        with pytest.raises(ValueError):
            verify(path)
