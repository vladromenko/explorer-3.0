"""Explicit one-time downloads; normal analysis remains strictly local."""
import argparse,json
from pathlib import Path
from explorer3.paths import ROOT,DATA
from explorer3.vision_assets import verify,digest


def main(method,download):
    root=DATA/'cache/vision';root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'configs/vision-assets.json').read_text())
    if download:
        if method=='sam3':
            from huggingface_hub import snapshot_download
            cfg=manifest['sam3']
            snapshot_download(cfg['repository'],revision=cfg['revision'],local_dir=root/'sam3',
                              allow_patterns=[x['file'] for x in cfg['files']])
        else:
            import requests
            for item in manifest['weights']:
                path=root/item['file']
                if not path.is_file() or digest(path)!=item['sha256']:
                    temporary=path.with_suffix('.download')
                    with requests.get(item['url'],stream=True,timeout=90) as response:
                        response.raise_for_status()
                        with temporary.open('wb') as output:
                            for chunk in response.iter_content(1024*1024):output.write(chunk)
                    if digest(temporary)!=item['sha256']:raise ValueError('Asset hash mismatch')
                    temporary.replace(path)
    print(json.dumps(verify(root,method),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=['yoloe','sam3']);parser.add_argument('--download',action='store_true')
    args=parser.parse_args();main(args.method,args.download)
