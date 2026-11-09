"""No network or model dependencies: exercise transport failures and restore safety."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/download_checkpoints.py'
spec=importlib.util.spec_from_file_location('checkpoint_download',SCRIPT)
downloader=importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)
CONTENT=b'synthetic checkpoint bytes for transfer verification'
FILE_ID='anonymousFileId_12345'


def entry(path='Results/3DINO/LinearProbe/CV/100_percent/fold_01/best_model.pt',**values):
    return {'path':path,'model':'3DINO','method':'LinearProbe','development_percentage':100,'partition':'fold_01','bytes':len(CONTENT),'sha256':hashlib.sha256(CONTENT).hexdigest(),'google_drive_file_id':FILE_ID,**values}


def manifest(tmp_path,entries=None):
    entries=entries or [entry()]
    path=tmp_path/'catalog.json'
    path.write_text(json.dumps({'checkpoint_count':len(entries),'bytes':sum(e['bytes'] for e in entries),'checkpoints':entries}))
    return path


def transfer(file_id,output,**kwargs):
    assert file_id==FILE_ID
    output.write_bytes(CONTENT)
    return str(output)


def test_default_lists_without_optional_packages_network_or_output_writes(tmp_path):
    source=manifest(tmp_path)
    output=tmp_path/'not-created'
    result=subprocess.run([sys.executable,'-S',str(SCRIPT),'--manifest',str(source),'--output-root',str(output)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'No downloads performed' in result.stdout
    assert not output.exists()


@pytest.mark.parametrize('unsafe',['../outside.pt','Results/../../outside.pt','/Results/a.pt','C:/Results/a.pt','Results\\outside.pt','Results/a/../outside.pt','Results//a.pt','Results/CON.pt','Results/stream:ads','Results/trailing.','Results/a?/b.pt'])
def test_manifest_rejects_cross_platform_unsafe_paths(tmp_path,unsafe):
    source=manifest(tmp_path,[entry(path=unsafe)])
    with pytest.raises(downloader.DownloadError,match='path|Results'):
        downloader.load_manifest(source)


def test_duplicate_paths_and_catalog_totals_rejected(tmp_path):
    source=manifest(tmp_path,[entry(),entry(path=entry()['path'].replace('3DINO','3dino'))])
    with pytest.raises(downloader.DownloadError,match='Duplicate'):
        downloader.load_manifest(source)
    source=manifest(tmp_path)
    data=json.loads(source.read_text())
    data['bytes']+=1
    source.write_text(json.dumps(data))
    with pytest.raises(downloader.DownloadError,match='total bytes'):
        downloader.load_manifest(source)


@pytest.mark.parametrize('url',[f'https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing',f'https://docs.google.com/uc?export=download&id={FILE_ID}',f'https://drive.google.com/open?id={FILE_ID}'])
def test_anonymous_public_file_urls_resolve_without_account(url):
    value=entry(google_drive_url=url)
    del value['google_drive_file_id']
    assert downloader.drive_file_id(value)==FILE_ID


def test_url_host_and_id_conflicts_fail_closed():
    with pytest.raises(downloader.DownloadError,match='disagree'):
        downloader.drive_file_id(entry(google_drive_url='https://drive.google.com/file/d/aDifferentId123/view'))
    with pytest.raises(downloader.DownloadError,match='HTTPS'):
        downloader.drive_file_id(entry(google_drive_url=f'https://drive.google.com.attacker.invalid/file/d/{FILE_ID}/view'))


def test_download_publishes_exact_path_and_skips_only_verified_existing_file(tmp_path):
    record=entry()
    target=tmp_path/record['path']
    assert downloader.download_one(record,tmp_path,transport=transfer)=='downloaded and verified'
    assert target.read_bytes()==CONTENT
    assert not (Path(str(target)+'.download')/'payload.part').exists()
    def forbidden(*a,**kw): raise AssertionError('Verified file must not download')
    assert downloader.download_one(record,tmp_path,transport=forbidden)=='already verified'
    target.write_bytes(b'conflicting final contents')
    with pytest.raises(downloader.DownloadError,match='Refusing to overwrite'):
        downloader.download_one(record,tmp_path,transport=forbidden)
    assert target.read_bytes()==b'conflicting final contents'


def test_hash_failure_preserves_final_path_and_requires_explicit_restart(tmp_path):
    record=entry()
    target=tmp_path/record['path']
    def corrupt(fid,output,**kwargs):
        output.write_bytes(b'x'*len(CONTENT))
        return str(output)
    with pytest.raises(downloader.DownloadError,match='SHA-256 mismatch'):
        downloader.download_one(record,tmp_path,transport=corrupt)
    assert not target.exists()
    with pytest.raises(downloader.DownloadError,match='restart'):
        downloader.download_one(record,tmp_path,transport=transfer)
    assert downloader.download_one(record,tmp_path,restart=True,transport=transfer)=='downloaded and verified'
    assert target.read_bytes()==CONTENT


def test_interrupted_transfer_resumes_owned_partial_and_releases_lock(tmp_path):
    record=entry()
    target=tmp_path/record['path']
    def interrupted(fid,output,**kwargs):
        output.with_name('payload.part123.part').write_bytes(CONTENT[:10])
        raise OSError('simulated network interruption')
    with pytest.raises(OSError,match='interruption'):
        downloader.download_one(record,tmp_path,transport=interrupted)
    assert not target.exists()
    def resumed(fid,output,**kwargs):
        partial=output.with_name('payload.part123.part')
        assert partial.read_bytes()==CONTENT[:10]
        with partial.open('ab') as handle: handle.write(CONTENT[10:])
        partial.rename(output)
        return str(output)
    assert downloader.download_one(record,tmp_path,transport=resumed)=='downloaded and verified'
    assert target.read_bytes()==CONTENT


def test_ignored_range_cannot_publish_corrupt_resumed_file(tmp_path):
    record=entry()
    def ignored_range(fid,output,**kwargs):
        output.write_bytes(CONTENT[:10]+CONTENT)
        return str(output)
    with pytest.raises(downloader.DownloadError,match='size mismatch'):
        downloader.download_one(record,tmp_path,transport=ignored_range)
    assert not (tmp_path/record['path']).exists()


def test_concurrent_differing_final_file_is_never_overwritten(tmp_path):
    record=entry()
    target=tmp_path/record['path']
    def race(fid,output,**kwargs):
        output.write_bytes(CONTENT)
        target.write_bytes(b'created by another process')
        return str(output)
    with pytest.raises(downloader.DownloadError,match='differing destination'):
        downloader.download_one(record,tmp_path,transport=race)
    assert target.read_bytes()==b'created by another process'


def test_staging_identity_and_unowned_files_are_not_adopted(tmp_path):
    record=entry()
    stage=Path(str(tmp_path/record['path'])+'.download')
    stage.mkdir(parents=True)
    unrelated=stage/'notes.txt'
    unrelated.write_text('keep me')
    with pytest.raises(downloader.DownloadError,match='Unrecognized'):
        downloader.download_one(record,tmp_path,restart=True,transport=transfer)
    assert unrelated.read_text()=='keep me'
    unrelated.unlink()
    (stage/'identity.json').write_text(json.dumps({'path':'different'}))
    with pytest.raises(downloader.DownloadError,match='identity differs'):
        downloader.download_one(record,tmp_path,restart=True,transport=transfer)


def test_symlink_escape_rejected_before_any_writes(tmp_path):
    root,outside=tmp_path/'output',tmp_path/'outside'
    root.mkdir(); outside.mkdir()
    try: (root/'Results').symlink_to(outside,target_is_directory=True)
    except OSError: pytest.skip('Symlink creation is not permitted on this platform')
    with pytest.raises(downloader.DownloadError,match='Symlink|escapes'):
        downloader.download_one(entry(),root,transport=transfer)
    assert not list(outside.iterdir())


def test_cli_filters_dry_run_and_verify_do_not_download(tmp_path,capsys,monkeypatch):
    source=manifest(tmp_path,[entry(),entry(path='Results/DenseNet/Scratch/20_percent/refit/best_model.pt',model='DenseNet',method='Scratch',development_percentage=20,partition='refit')])
    def forbidden(*a,**kw): raise AssertionError('No network expected')
    monkeypatch.setattr(downloader,'anonymous_download',forbidden)
    argv=['--manifest',str(source),'--output-root',str(tmp_path/'output')]
    assert downloader.main(argv+['--model','threedino','--percentage','100','--dry-run'])==0
    text=capsys.readouterr().out
    assert '1 checkpoints;' in text and 'Results/DenseNet' not in text
    assert not (tmp_path/'output').exists()
    assert downloader.main(argv+['--verify-only'])==1
    downloader.download_one(entry(),tmp_path/'output',transport=transfer)
    assert downloader.main(argv+['--verify-only','--path',entry()['path']])==0
    assert downloader.main(argv+['--all','--model','DenseNet'])==2


def test_lazy_gdown_adapter_is_anonymous_and_uses_safe_tls_resume(tmp_path,monkeypatch):
    observed={}
    def fake(**kwargs): observed.update(kwargs); return kwargs['output']
    monkeypatch.setitem(sys.modules,'gdown',SimpleNamespace(download=fake))
    output=tmp_path/'payload.part'
    assert downloader.anonymous_download(FILE_ID,output,quiet=True)==str(output)
    assert observed=={'id':FILE_ID,'output':str(output),'resume':True,'use_cookies':False,'quiet':True,'verify':True}


def test_missing_optional_downloader_has_clear_install_instruction(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'gdown',None)
    with pytest.raises(downloader.DownloadError,match='pip install -r scripts/requirements-download.txt'):
        downloader.anonymous_download(FILE_ID,tmp_path/'payload.part')


def test_same_staging_directory_cannot_be_used_by_two_writers(tmp_path):
    stage=tmp_path/'download'
    stage.mkdir()
    with downloader.stage_lock(stage):
        with pytest.raises(downloader.DownloadError,match='Another download'):
            with downloader.stage_lock(stage):
                raise AssertionError('A concurrent writer obtained the staging lock')
    with downloader.stage_lock(stage):
        pass
