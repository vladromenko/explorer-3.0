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


def test_install_backup_and_rollback_preserve_secrets_and_legacy_history(tmp_path):
    import hashlib,json
    from scripts.install_tank_bundle import install,rollback
    root=tmp_path/'robot';root.mkdir();stage=tmp_path/'stage';(stage/'payload').mkdir(parents=True)
    (root/'agent.py').write_text('old agent');(root/'config.json').write_text('secret remains')
    (root/'incident-block.json').write_text('old incident')
    files={'agent.py':b'new agent','limits.py':b'new limits'}
    for name,data in files.items():(stage/'payload'/name).write_bytes(data)
    manifest={'platform':'tank','files':{'payload/'+name:hashlib.sha256(data).hexdigest() for name,data in files.items()}}
    (stage/'MANIFEST.json').write_text(json.dumps(manifest))
    result=install(stage,root)
    assert not (root/'incident-block.json').exists() and (root/'config.json').read_text()=='secret remains'
    assert (root/'agent.py').read_text()=='new agent'
    rollback(result['archive'])
    assert (root/'agent.py').read_text()=='old agent' and not (root/'limits.py').exists()
    assert (root/'incident-block.json').read_text()=='old incident'
    (stage/'payload/agent.py').write_text('tampered')
    with pytest.raises(ValueError,match='hash'):install(stage,root)
