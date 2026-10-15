"""Select using January validation only, then evaluate frozen datasets."""
from collections import Counter
import gc
import hashlib
import time
import numpy as np
import torch
from bert_punct import (ROOT, IGNORE, LABELS, Predictor, cm_update, evaluate_predictions,
                        jsonl, metrics, read_json, render, write_json, write_jsonl)

OUT=ROOT/'runs/v2_evaluation'


def aggregate(rows,predictions,cfg):
    result=evaluate_predictions(rows,predictions,cfg)
    counts=[]
    exact=0
    for row,pred in zip(rows,predictions):
        gold=np.array(row['labels'])
        valid=gold!=IGNORE
        exact+=int(np.all(gold[valid]==pred[valid]))
        counts.append([int(np.sum(valid&(gold>0)&(pred==gold))),int(np.sum(valid&(gold>0))),int(np.sum(valid&(pred>0)))])
    result['article_exact_match']=exact/len(rows)
    result['articles']=len(rows)
    result['characters']=sum(len(r['text']) for r in rows)
    return result,np.array(counts,dtype=np.int64)


def bootstrap(counts,other=None,repeats=1000):
    rng=np.random.default_rng(20261015)
    samples=[]
    for _ in range(repeats):
        index=rng.integers(len(counts),size=len(counts))
        a=counts[index].sum(0)
        value=2*a[0]/max(1,a[1]+a[2])
        if other is not None:
            b=other[index].sum(0)
            value-=2*b[0]/max(1,b[1]+b[2])
        samples.append(value)
    return {'lower':float(np.percentile(samples,2.5)),'upper':float(np.percentile(samples,97.5)),
            'replicates':repeats,'unit':'whole article (classical: chapter)','seed':20261015}


def save_predictions(path,rows,predictions):
    write_jsonl(path,[{'id':r['id'],'prediction':p.tolist()} for r,p in zip(rows,predictions)])


def main():
    candidates=['corrected_seed42','corrected_seed43','corrected_seed44','weighted_seed42']
    details={}
    for name in candidates:
        cfg=read_json(ROOT/f'configs/{name}.json')
        completed=read_json(ROOT/cfg['run_dir']/'completed.json')
        assert completed['status']=='completed' and completed['epochs']==3
        selection=read_json(ROOT/cfg['run_dir']/'best/selection.json')
        details[name]={'config':cfg,'selection':selection,'completed':completed}
    # Seeds 43/44 are replication runs, never candidates for choosing a lucky seed.
    chosen=max(['corrected_seed42','weighted_seed42'],key=lambda n:details[n]['selection']['validation_micro_f1'])
    cfg={**details[chosen]['config'],'checkpoint':details[chosen]['config']['run_dir']+'/best'}
    final_selection={'chosen':chosen,'criterion':'max January validation punctuation micro-F1; seed fixed to 42; no test-based choice',
        'candidates':details,'chosen_before_test_evaluation':True,'default_decoder':'fast (same labels as fixed)'}
    write_json(OUT/'selection.json',final_selection)
    write_json(ROOT/'config.json',cfg)
    datasets={name:jsonl(ROOT/f'data/v2/{file}.jsonl') for name,file in
              [('january','test'),('february','february_holdout'),('classical','classical_reference')]}
    results={}
    counts_by_method={}
    for name in candidates:
        model_cfg=details[name]['config']
        predictor=Predictor(model_cfg,ROOT/model_cfg['run_dir']/'best')
        results[name]={}
        for dataset,rows in datasets.items():
            start=time.perf_counter()
            pred=predictor.fixed_fast([r['text'] for r in rows],16)
            metric,counts=aggregate(rows,pred,model_cfg)
            metric['elapsed_s_unwarmed']=time.perf_counter()-start
            metric['micro_f1_ci95']=bootstrap(counts)
            results[name][dataset]=metric
            if name==chosen:
                counts_by_method[dataset]=counts
            save_predictions(OUT/f'{name}/{dataset}.jsonl',rows,pred)
            print(name,dataset,'micro',metric['overall']['punctuation_micro_f1'],flush=True)
            write_json(OUT/'model_metrics.json',results)
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
    predictor=Predictor(cfg)
    methods={}
    equivalence={}
    for dataset in ['january','classical']:
        rows=datasets[dataset]
        texts=[r['text'] for r in rows]
        methods[dataset]={}
        fast=np.concatenate([np.array(r['prediction']) for r in jsonl(OUT/f'{chosen}/{dataset}.jsonl')])
        for method in ['all_O','fixed','sequential','fusion','nonoverlap']:
            if method=='all_O':
                pred=[np.zeros(len(t),dtype=np.int64) for t in texts]
            elif method=='sequential':
                pred=predictor.sequential(texts)
            elif method=='nonoverlap':
                predictor.cfg={**cfg,'window':300,'stride':300}
                pred=predictor.fixed_fast(texts,16)
                predictor.cfg=cfg
            else:
                pred=predictor.fixed(texts,16,fusion=method=='fusion')
            metric,counts=aggregate(rows,pred,cfg)
            metric['paired_micro_delta_vs_fast_ci95']=bootstrap(counts,counts_by_method[dataset])
            methods[dataset][method]=metric
            save_predictions(OUT/f'methods/{dataset}_{method}.jsonl',rows,pred)
            if method=='fixed':
                differences=int(np.count_nonzero(fast!=np.concatenate(pred)))
                equivalence[dataset]={'fast_vs_reference_different_positions':differences,'characters':len(fast)}
                assert differences==0,'Fast scatter changes predicted labels'
            print('method',dataset,method,metric['overall']['punctuation_micro_f1'],flush=True)
        write_json(OUT/'method_metrics.json',methods)
    # Historical January comparison is retrospective; never call it a new blind test.
    historical_path=ROOT/'runs/main/test/fixed_predictions.jsonl'
    if historical_path.exists():
        old=jsonl(historical_path)
        assert [r['id'] for r in old]==[r['id'] for r in datasets['january']]
        old_metric,old_counts=aggregate(datasets['january'],[np.array(r['prediction']) for r in old],cfg)
        write_json(OUT/'historical_comparison.json',{'old_v1':old_metric,
            'paired_micro_delta_new_minus_old_ci95':bootstrap(counts_by_method['january'],old_counts),
            'caveat':'Old January test already observed; new February set is the frozen final test. Training budgets differ for this historical comparison.'})
    write_json(OUT/'decoder_equivalence.json',equivalence)
    errors=[]
    rows=datasets['february']
    pred=jsonl(OUT/f'{chosen}/february.jsonl')
    seen=Counter()
    for row,out in zip(rows,pred):
        for i,(g,p) in enumerate(zip(row['labels'],out['prediction'])):
            if g==IGNORE or g==p or seen[g,p]>=2: continue
            seen[g,p]+=1
            lo,hi=max(0,i-30),min(len(row['text']),i+31)
            errors.append({'id':row['id'],'position':i,'gold':LABELS[g],'prediction':LABELS[p],
                'reference_context':render(row['text'][lo:hi],row['labels'][lo:hi]),
                'predicted_context':render(row['text'][lo:hi],out['prediction'][lo:hi])})
    write_jsonl(OUT/'february_errors.jsonl',errors)
    historical_examples=ROOT/'runs/main/test/classical_examples.json'
    old_examples=read_json(historical_examples if historical_examples.exists() else ROOT/'examples/classical.json')
    example_preds=predictor.fixed_fast([r['text'] for r in old_examples],16)
    for row,p in zip(old_examples,example_preds):
        if 'prediction' in row:
            row['v1_prediction']=row.pop('prediction')
        row['v2_prediction']=render(row['text'],p)
    write_json(OUT/'classical_examples.json',old_examples)
    # Matrix of per-class counts/uncertainties above is preserved, including zeros.
    write_json(OUT/'completed.json',{'status':'completed','chosen':chosen,'models':candidates,
               'datasets':{k:len(v) for k,v in datasets.items()},'predictions_saved':True})


if __name__=='__main__':
    main()
