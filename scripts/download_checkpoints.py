#!/usr/bin/env python3
"""List, verify, or restore published experiment checkpoints from public Drive.

No arguments lists the catalog. Downloading requires --all or an explicit filter.
Interrupted transfers resume in an owned staging directory. Every final file is
size/SHA-256 verified; existing differing files are never replaced. No PyTorch or
Google account is required. Install requirements-download.txt only to download.
Publication requires a filesystem supporting hard links (for example NTFS/ext4).
Each *.download staging directory retains a small identity record and lock file;
only transfer .part files consume checkpoint storage. --restart resets owned parts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
ID_PATTERN = re.compile(r'[A-Za-z0-9_-]{10,}')
SHA_PATTERN = re.compile(r'[0-9a-fA-F]{64}')
WINDOWS_RESERVED = {'CON','PRN','AUX','NUL'} | {f'{p}{n}' for p in ('COM','LPT') for n in range(1,10)}


class DownloadError(ValueError):
    """An invalid catalog, selection, destination, or transfer."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative_parts(value):
    if not isinstance(value,str) or '\\' in value:
        raise DownloadError(f'Expected a portable Results/... path: {value!r}')
    parts = value.split('/')
    if len(parts)<2 or parts[0]!='Results':
        raise DownloadError(f'Checkpoint path must be under Results/: {value!r}')
    for part in parts:
        if part in ('','.','..') or part.endswith(('.', ' ')) or any(ord(c)<32 or c in '<>:"|?*' for c in part) or part.split('.')[0].upper() in WINDOWS_RESERVED:
            raise DownloadError(f'Unsafe checkpoint path: {value!r}')
    return parts


def safe_target(root, relative):
    root = Path(root).resolve()
    target = root
    for part in relative_parts(relative):
        target = target / part
        if target.is_symlink() or (hasattr(target,'is_junction') and target.is_junction()):
            raise DownloadError(f'Symlink/junction in destination path: {target}')
        if not target.resolve().is_relative_to(root):
            raise DownloadError(f'Destination escapes output root: {target}')
    return target


def drive_file_id(entry):
    direct = entry.get('google_drive_file_id')
    if direct is not None and (not isinstance(direct,str) or not ID_PATTERN.fullmatch(direct)):
        raise DownloadError(f'Invalid google_drive_file_id for {entry["path"]}')
    url = entry.get('google_drive_url')
    extracted = None
    if url:
        if not isinstance(url,str):
            raise DownloadError('google_drive_url must be a string')
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError as exc:
            raise DownloadError(f'Malformed Google Drive URL: {url}') from exc
        if parsed.scheme != 'https' or parsed.hostname not in ('drive.google.com','docs.google.com') or parsed.username or parsed.password or port not in (None,443):
            raise DownloadError(f'Expected a public Google Drive HTTPS file URL: {url}')
        match = re.fullmatch(r'/file/d/([A-Za-z0-9_-]+)(?:/(?:view|edit|preview))?/?', parsed.path)
        if match:
            extracted = match.group(1)
        elif parsed.path in ('/uc','/open','/download'):
            ids = parse_qs(parsed.query).get('id',[])
            if len(ids)==1: extracted=ids[0]
        if not extracted or not ID_PATTERN.fullmatch(extracted):
            raise DownloadError(f'Cannot identify a binary file in Google Drive URL: {url}')
    if direct and extracted and direct != extracted:
        raise DownloadError(f'Google Drive file ID and URL disagree: {entry["path"]}')
    if not (direct or extracted):
        raise DownloadError(f'No public Google Drive file ID/URL recorded for {entry["path"]}')
    return direct or extracted


def load_manifest(path):
    try:
        document=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError,json.JSONDecodeError) as exc:
        raise DownloadError(f'Cannot read checkpoint manifest {path}: {exc}') from exc
    entries=document.get('checkpoints') if isinstance(document,dict) else None
    if not isinstance(entries,list) or not entries:
        raise DownloadError('Manifest must contain a nonempty checkpoints list')
    seen=set()
    for entry in entries:
        if not isinstance(entry,dict): raise DownloadError('Each checkpoint must be an object')
        relative_parts(entry.get('path'))
        key=entry['path'].casefold()
        if key in seen: raise DownloadError(f'Duplicate checkpoint path: {entry["path"]}')
        seen.add(key)
        if type(entry.get('bytes')) is not int or entry['bytes']<=0:
            raise DownloadError(f'Invalid checkpoint size: {entry["path"]}')
        if not isinstance(entry.get('sha256'),str) or not SHA_PATTERN.fullmatch(entry['sha256']):
            raise DownloadError(f'Invalid SHA-256: {entry["path"]}')
        for name in ('model','method','partition'):
            if not isinstance(entry.get(name),str) or not entry[name]:
                raise DownloadError(f'Missing {name}: {entry["path"]}')
        if type(entry.get('development_percentage')) is not int or not 0 < entry['development_percentage'] <= 100:
            raise DownloadError(f'Invalid development_percentage: {entry["path"]}')
    if 'checkpoint_count' in document and document['checkpoint_count'] != len(entries):
        raise DownloadError('Manifest checkpoint_count differs from its entries')
    if 'bytes' in document and document['bytes'] != sum(e['bytes'] for e in entries):
        raise DownloadError('Manifest total bytes differs from its entries')
    return document,entries


def normalized_model(value):
    aliases={'threedino':'3dino','pillar-0':'pillar0'}
    return aliases.get(value.casefold(),value.casefold())


def selected_entries(entries,args):
    filters=bool(args.model or args.method or args.percentage or args.partition or args.path)
    if args.all and filters:
        raise DownloadError('Use --all or filters, not both')
    paths=set(args.path or [])
    for path in paths: relative_parts(path)
    if paths-set(e['path'] for e in entries):
        raise DownloadError('At least one --path is absent from the catalog')
    result=[e for e in entries if
            (not args.model or normalized_model(e['model']) in {normalized_model(x) for x in args.model}) and
            (not args.method or e['method'].casefold() in {x.casefold() for x in args.method}) and
            (not args.percentage or e['development_percentage'] in args.percentage) and
            (not args.partition or e['partition'].casefold() in {x.casefold() for x in args.partition}) and
            (not paths or e['path'] in paths)]
    if not result: raise DownloadError('No checkpoints match all filters')
    return result,filters


def verify_file(path,entry):
    path=Path(path)
    if not path.exists(): return 'missing'
    if not path.is_file(): return 'not a regular file'
    if path.stat().st_size != entry['bytes']: return 'size mismatch'
    if sha256(path).lower()!=entry['sha256'].lower(): return 'SHA-256 mismatch'
    return 'verified'


@contextmanager
def stage_lock(stage):
    """OS-held lock; released automatically even if a process is terminated."""
    lock=stage/'.lock'
    if lock.is_symlink(): raise DownloadError(f'Unsafe lock path: {lock}')
    with lock.open('a+b') as handle:
        if handle.tell()==0: handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            raise DownloadError(f'Another download is using {stage}') from exc
        try: yield
        finally:
            handle.seek(0)
            if os.name=='nt': msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def anonymous_download(file_id,output,*,quiet=False):
    try:
        import gdown
    except ImportError as exc:
        raise DownloadError('Downloading requires gdown. Install with: python -m pip install -r scripts/requirements-download.txt') from exc
    # No user account/cookies. gdown handles Google's large-file confirmation.
    try:
        return gdown.download(id=file_id,output=str(output),resume=True,use_cookies=False,quiet=quiet,verify=True)
    except Exception as exc:
        raise DownloadError(f'Public Google Drive transfer failed for {file_id}: {exc}. Partial files retained; repeat the selection to resume.') from exc


def download_one(entry,root,*,restart=False,quiet=False,transport=None):
    target=safe_target(root,entry['path'])
    status=verify_file(target,entry)
    if status=='verified': return 'already verified'
    if status!='missing':
        raise DownloadError(f'Refusing to overwrite existing {target}: {status}. Move it aside explicitly first.')
    file_id=drive_file_id(entry)
    stage=safe_target(root,entry['path']+'.download')
    target.parent.mkdir(parents=True,exist_ok=True)
    stage.mkdir(exist_ok=True)
    identity={'schema_version':1,'path':entry['path'],'bytes':entry['bytes'],'sha256':entry['sha256'].lower(),'google_drive_file_id':file_id}
    with stage_lock(stage):
        marker=stage/'identity.json'
        for item in stage.iterdir():
            if item.is_symlink() or not item.is_file() or (item.name not in ('.lock','identity.json') and not (item.name.startswith('payload.part') and item.name.endswith('.part'))):
                raise DownloadError(f'Unrecognized download staging content: {item}')
        if marker.exists():
            try: previous=json.loads(marker.read_text(encoding='utf-8'))
            except (OSError,json.JSONDecodeError) as exc: raise DownloadError(f'Invalid staging identity: {marker}') from exc
            if previous != identity:
                raise DownloadError(f'Staging identity differs from catalog: {stage}; move the staging directory aside before retrying')
        else:
            if any(item.name!='.lock' for item in stage.iterdir()):
                raise DownloadError(f'Unowned partial download in {stage}; move it aside before retrying')
            with marker.open('x',encoding='utf-8') as handle:
                json.dump(identity,handle,sort_keys=True)
        output=stage/'payload.part'
        if restart:
            for item in stage.iterdir():
                if item.name.startswith('payload.part') and item.name.endswith('.part'): item.unlink()
        elif output.exists() and verify_file(output,entry)!='verified':
            raise DownloadError(f'Completed temporary transfer failed verification: {output}; retry with --restart')
        if not output.exists():
            download=transport or anonymous_download
            result=download(file_id,output,quiet=quiet)
            if result is None:
                raise DownloadError(f'Download did not complete: {entry["path"]}. Retry or use --restart for conflicting partial transfers.')
        if output.is_symlink(): raise DownloadError(f'Unsafe downloaded output: {output}')
        status=verify_file(output,entry)
        if status!='verified':
            raise DownloadError(f'Download verification failed ({status}): {entry["path"]}; final path untouched. Retry with --restart.')
        with output.open('r+b') as handle: os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if a final path appeared
        # concurrently. Unlike replace()/POSIX rename(), it cannot overwrite it.
        safe_target(root,entry['path'])
        try:
            os.link(output,target)
        except FileExistsError:
            if verify_file(target,entry)!='verified':
                raise DownloadError(f'A differing destination appeared during download: {target}')
        except OSError as exc:
            raise DownloadError(f'Cannot atomically publish {target}; use a filesystem supporting hard links. Verified temporary file retained: {output}. {exc}') from exc
        output.unlink()
        return 'downloaded and verified'


def parser():
    result=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument('--manifest',type=Path,default=REPO_ROOT/'provenance/checkpoints.json')
    result.add_argument('--output-root',type=Path,default=REPO_ROOT,help='Restore exact Results paths below this directory')
    mode=result.add_mutually_exclusive_group()
    mode.add_argument('--list',action='store_true',help='List selected checkpoints without downloading (default without filters)')
    mode.add_argument('--verify-only',action='store_true',help='Read local files and verify size/SHA-256; no network')
    result.add_argument('--all',action='store_true',help='Explicitly select the full catalog for downloading')
    result.add_argument('--model',action='append',help='Model name; repeat for OR selection (3DINO/ThreeDINO accepted)')
    result.add_argument('--method',action='append',help='Method name; repeat for OR selection')
    result.add_argument('--percentage',action='append',type=int,help='Development percentage; repeatable')
    result.add_argument('--partition',action='append',help='Partition such as fold_01; repeatable')
    result.add_argument('--path',action='append',help='Exact catalog Results/... path; repeatable')
    result.add_argument('--dry-run',action='store_true',help='List selection and total bytes without writing or networking')
    result.add_argument('--restart',action='store_true',help='Discard only owned temporary transfer files before downloading again')
    result.add_argument('--quiet',action='store_true',help='Suppress transport progress bars')
    return result


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        document,entries=load_manifest(args.manifest)
        chosen,filters=selected_entries(entries,args)
        listing=args.list or args.dry_run or (not args.all and not filters and not args.verify_only)
        if args.restart and (listing or args.verify_only):
            raise DownloadError('--restart requires a download selection (--all or filters)')
        if listing:
            for entry in chosen:
                print(f'{entry["bytes"]:>12}  {entry["path"]}')
            print(f'{len(chosen)} checkpoints; {sum(e["bytes"] for e in chosen):,} bytes total. No downloads performed.')
            if document.get('drive_folder_url'): print(f'Public folder: {document["drive_folder_url"]}')
            return 0
        failed=False
        # Preflight every final destination before the first network request.
        for entry in chosen:
            target=safe_target(args.output_root,entry['path'])
            status=verify_file(target,entry)
            if args.verify_only:
                print(f'{status}: {entry["path"]}')
                failed |= status!='verified'
            elif status not in ('verified','missing'):
                raise DownloadError(f'Refusing to overwrite existing {target}: {status}. Move it aside explicitly first.')
            elif status=='missing': drive_file_id(entry)
        if args.verify_only: return 1 if failed else 0
        print(f'Selected {len(chosen)} checkpoints; {sum(e["bytes"] for e in chosen):,} bytes before verified-file skips.')
        for entry in chosen:
            status=download_one(entry,args.output_root,restart=args.restart,quiet=args.quiet)
            print(f'{status}: {entry["path"]}')
        return 0
    except (DownloadError,OSError) as exc:
        print(f'error: {exc}',file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('Interrupted; partial transfers retained. Repeat the same selection to resume.',file=sys.stderr)
        return 130


if __name__=='__main__':
    raise SystemExit(main())


