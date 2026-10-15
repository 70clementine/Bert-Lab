"""Audit v2 against the OFFICIAL vocabulary and independently saved predictions."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import unicodedata

ROOT=Path(__file__).resolve().parent


def load(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def rows(path):
    with (ROOT/path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    checks=[]
    vocab=load('models/bert-base-chinese/tokenizer.json')['model']['vocab']
    with (ROOT/'models/bert-base-chinese/vocab.txt').open(encoding='utf-8') as f:
        parsed={line.rstrip('\n'):i for i,line in enumerate(f)}
    assert parsed==vocab and len(vocab)==21128 and vocab['我']==2769 and vocab['\u2028']==343
    checks.append('All 21,128 vocabulary entries match tokenizer.json; U+2028 retained at 343; 我=2769')
    splits={name:rows(f'data/v2/{name}.jsonl') for name in ['train','val','test','february_holdout','classical_reference']}
    for name,data in splits.items():
        for r in data:
            assert len(r['text'])==len(r['labels'])==len(r['input_ids'])
            expected=[]
            for c in r['text']:
                normalized=unicodedata.normalize('NFKC',c).lower()
                token=normalized if len(normalized)==1 else c
                expected.append(vocab.get(token,vocab.get(c,vocab['[UNK]'])))
            assert expected==r['input_ids'],r['id']
        checks.append(f'{name}: every cached character ID checked against canonical vocabulary ({len(data)} documents)')
    reference='archives/v1_20261014/processed' if (ROOT/'archives/v1_20261014/processed').exists() else 'data/processed'
    for name in ['train','val','test']:
        old=rows(f'{reference}/{name}.jsonl')
        assert len(old)==len(splits[name])
        assert all(all(a[k]==b[k] for k in ['id','text','labels','hash']) for a,b in zip(old,splits[name]))
    checks.append('January IDs/splits/text/labels match the saved preparation reference: '+reference)
    names=['train','val','test','february_holdout']
    for i,a in enumerate(names):
        for b in names[i+1:]:
            assert not ({r['hash'] for r in splits[a]} & {r['hash'] for r in splits[b]})
            assert not ({h for r in splits[a] for h in r.get('paragraph_hashes',[])} & {h for r in splits[b] for h in r.get('paragraph_hashes',[])})
    checks.append('January train/val/test and February: no exact article or >=40-character exact paragraph overlap')
    selection=load('runs/v2_evaluation/selection.json')
    recipes=selection['candidates']
    for name,info in recipes.items():
        cfg=info['config']
        history=load(cfg['run_dir']+'/history.json')
        done=load(cfg['run_dir']+'/completed.json')
        assert len(history)==done['epochs']==cfg['epochs']==3
        assert done['optimizer_updates']==1005
        best=max(history,key=lambda h:h['validation']['punctuation_micro_f1'])
        assert best['epoch']==info['selection']['epoch']
        modelcfg=load(cfg['run_dir']+'/best/config.json')
        assert modelcfg['char_vocab_reader']=='lf_v2' and modelcfg['vocab_size']==21128
        checks.append(f'{name}: 3 complete epochs / 1005 updates; validation-selected checkpoint declares lf_v2')
    expected=max(['corrected_seed42','weighted_seed42'],key=lambda n:recipes[n]['selection']['validation_micro_f1'])
    assert selection['chosen']==expected
    a,b=recipes['corrected_seed42']['config'],recipes['weighted_seed42']['config']
    for field in ['seed','model_id','model_revision','epochs','learning_rate','train_batch_size','gradient_accumulation','weight_decay','data_dir','window','stride']:
        assert a[field]==b[field],field
    checks.append('Class weighting comparison uses the same seed, initialization, data, learning rate and update budget; selection excludes replication seeds')
    def verify_prediction(data,outputs,metric):
        assert len(data)==len(outputs)
        counts=Counter()
        for r,o in zip(data,outputs):
            assert r['id']==o['id'] and len(r['text'])==len(o['prediction'])
            assert all(0<=p<8 for p in o['prediction'])
            counts.update((g,p) for g,p in zip(r['labels'],o['prediction']) if g!=-100)
        matrix=[[counts[i,j] for j in range(8)] for i in range(8)]
        assert matrix==metric['overall']['confusion_matrix']
        tp=sum(n for (g,p),n in counts.items() if g==p and g>0)
        gold=sum(n for (g,p),n in counts.items() if g>0)
        predicted=sum(n for (g,p),n in counts.items() if p>0)
        assert abs(2*tp/max(1,gold+predicted)-metric['overall']['punctuation_micro_f1'])<1e-12
    mapping={'january':'test','february':'february_holdout','classical':'classical_reference'}
    for model,ds in load('runs/v2_evaluation/model_metrics.json').items():
        for dataset,metric in ds.items():
            verify_prediction(splits[mapping[dataset]],rows(f'runs/v2_evaluation/{model}/{dataset}.jsonl'),metric)
            checks.append(f'Independent confusion/F1 recomputation: {model}/{dataset} passed')
    for dataset,methods in load('runs/v2_evaluation/method_metrics.json').items():
        for method,metric in methods.items():
            verify_prediction(splits[mapping[dataset]],rows(f'runs/v2_evaluation/methods/{dataset}_{method}.jsonl'),metric)
    for metric in load('runs/v2_evaluation/decoder_equivalence.json').values():
        assert metric['fast_vs_reference_different_positions']==0
    checks.append('Every ablation prediction audited; fast/reference exactly agree on full January test and classical reference')
    edge=load('artifacts/v2/gpu-edge-tests.json')
    assert edge['status']=='passed' and edge['checkpoint']==load('config.json')['checkpoint']
    checks.append('GPU boundary integration: empty inputs, mixed lengths, tail boundaries, unknowns and invalid batch passed')
    raw=rows('runs/v2_benchmark/raw.jsonl')
    repeats=load('runs/v2_benchmark/manifest.json')['repeats']
    summaries=load('runs/v2_benchmark/summary.json')
    assert len(summaries)==33 and len(raw)==33*repeats
    assert all(r['status']=='ok' for r in raw)
    assert all(r['different_positions_vs_fixed16']==0 for r in load('runs/v2_benchmark/equivalence.json'))
    for item in summaries:
        samples=[r for r in raw if r['status']=='ok' and all(r[k]==item[k] for k in ['workload','method','batch'])]
        assert len(samples)==repeats
        assert statistics.median(r['elapsed_s'] for r in samples)==item['median_s']
        assert all(r['elapsed_s']>=r['model_s'] for r in samples)
    checks.append(f'Benchmark: {len(summaries)} configurations, {repeats} repetitions each; medians and CUDA timing bounds verified')
    status={'status':'passed','checks':checks,'scope':'canonical IDs, split integrity, training budgets, saved predictions, timing arithmetic; not a proof of universal correctness'}
    (ROOT/'artifacts/v2/final-audit.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n'.join(checks))
    print('V2 FINAL AUDIT PASSED')


if __name__=='__main__':
    main()
