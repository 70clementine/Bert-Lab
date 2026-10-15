"""Download only the pinned BERT files needed by this project, with integrity log."""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def main():
    cfg = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    dest = ROOT / cfg['model_dir']
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {'model_id': cfg['model_id'], 'revision': cfg['model_revision'], 'files': {}}
    for name in ['config.json', 'vocab.txt', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors']:
        path = dest / name
        url = f"https://huggingface.co/{cfg['model_id']}/resolve/{cfg['model_revision']}/{name}"
        if not path.exists():
            tmp = path.with_suffix(path.suffix + '.part')
            for attempt in range(8):
                try:
                    print('Downloading', name, flush=True)
                    size = tmp.stat().st_size if tmp.exists() else 0
                    headers = {'User-Agent': 'bertLab-course/1.0'}
                    if size:
                        headers['Range'] = f'bytes={size}-'
                    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                        if size and response.status != 206:
                            size = 0
                        mode = 'ab' if size else 'wb'
                        with tmp.open(mode) as out:
                            last = time.time()
                            while chunk := response.read(1024 * 1024):
                                out.write(chunk)
                                size += len(chunk)
                                if time.time() - last > 20:
                                    out.flush()
                                    print(name, round(size / 1024**2, 1), 'MiB', flush=True)
                                    last = time.time()
                    if name == 'model.safetensors':
                        expected = '3404a1ffd8da507042e8161013ba2a4fc49858b4e3f8fbf5ce5724f94883aec3'
                        assert hashlib.sha256(tmp.read_bytes()).hexdigest() == expected, 'BERT weight checksum mismatch'
                    os.replace(tmp, path)
                    break
                except Exception as exc:
                    print('Retry:', repr(exc), flush=True)
                    if attempt == 7:
                        raise
                    time.sleep(2)
        manifest['files'][name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'url': url}
    (dest / 'download_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Model download complete.', flush=True)


if __name__ == '__main__':
    main()
