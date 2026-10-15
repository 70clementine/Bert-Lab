"""Run predeclared reproducibility and class-weight experiments sequentially."""
import os
import subprocess
import sys
from bert_punct import ROOT, read_json


def main():
    (ROOT/'work').mkdir(parents=True,exist_ok=True)
    for name in ['corrected_seed43','corrected_seed44','weighted_seed42']:
        cfg=read_json(ROOT/f'configs/{name}.json')
        completion=ROOT/cfg['run_dir']/'completed.json'
        if completion.exists():
            assert read_json(completion)['status']=='completed'
            print('Already complete:',name,flush=True)
            continue
        print('START',name,flush=True)
        with (ROOT/f'work/v2-{name}.log').open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-u','bert_punct.py','train','--config',f'configs/{name}.json'],
                           cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,
                           env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        print('COMPLETE',name,read_json(completion),flush=True)


if __name__=='__main__':
    main()
