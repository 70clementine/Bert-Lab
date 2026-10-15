"""Same-checkpoint, repeated synchronized timing with algorithm/implementation isolation."""
import gc
import argparse
import random
import statistics
import time
import numpy as np
import torch
from bert_punct import ROOT, Predictor, jsonl, read_json, write_json, write_jsonl


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=7)
    args=parser.parse_args()
    if args.repeats<3: parser.error('At least three repetitions are required.')
    cfg=read_json(ROOT/'config.json')
    selected=read_json(ROOT/'runs/v2_evaluation/selection.json')
    assert cfg['checkpoint']==selected['candidates'][selected['chosen']]['config']['run_dir']+'/best'
    predictor=Predictor(cfg)
    rows=sorted(jsonl(ROOT/'data/v2/test.jsonl'),key=lambda r:len(r['text']),reverse=True)
    multi=[r['text'][:6000] for r in rows[:12]]
    assembled=''.join(r['text'] for r in rows)[:100000]
    workloads={'single_4020':[rows[0]['text'][:12000]],'multi_12':multi,'assembled_100k':[assembled]}
    methods=[('sequential',1),('fixed',1),('fixed',16),('fixed',64)]+[('fast',b) for b in [1,2,4,8,16,32,64]]
    repeats=args.repeats
    out=ROOT/'runs/v2_benchmark'
    manifest={'checkpoint':cfg['checkpoint'],'precision':cfg['precision'],'repeats':repeats,
        'workloads':{k:{'documents':len(v),'characters':sum(map(len,v)),'lengths':list(map(len,v))} for k,v in workloads.items()},
        'source_ids':[r['id'] for r in rows[:12]],
        'assembled_100k_note':'Concatenation of held-out January articles for performance stress ONLY; artificial joins are not quality-evaluated',
        'timing':'Warm model; encode, collate, H2D, forward, D2H, merge included; no disk or weight loading; synchronized boundaries',
        'confidence_note':'Median/min/max and paired speed ratios; variability across runs, not hardware-population inference'}
    write_json(out/'manifest.json',manifest)
    raw=[]
    equivalence=[]
    for workload,texts in workloads.items():
        reference=predictor.fixed(texts,16)
        allowed=[]
        for method,batch in methods:
            try:
                for _ in range(2):
                    if method=='sequential': predictor.sequential([texts[0][:1200]])
                    elif method=='fixed': predictor.fixed([texts[0][:400]]*batch,batch)
                    else: predictor.fixed_fast([texts[0][:400]]*batch,batch)
                allowed.append((method,batch))
            except torch.OutOfMemoryError:
                raw.append({'workload':workload,'method':method,'batch':batch,'status':'OOM warmup'})
                gc.collect(); torch.cuda.empty_cache()
        for repeat in range(repeats):
            order=allowed.copy()
            random.Random(1015+repeat).shuffle(order)
            for method,batch in order:
                try:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter()
                    if method=='sequential': pred=predictor.sequential(texts)
                    elif method=='fixed': pred=predictor.fixed(texts,batch)
                    else: pred=predictor.fixed_fast(texts,batch)
                    torch.cuda.synchronize()
                    elapsed=time.perf_counter()-start
                    row={'workload':workload,'method':method,'batch':batch,'repeat':repeat,'status':'ok',
                        'elapsed_s':elapsed,'chars_per_s':sum(map(len,texts))/elapsed,
                        'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                        'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,**predictor.last_stats}
                    raw.append(row)
                    if repeat==0 and method!='sequential':
                        diff=sum(int(np.count_nonzero(a!=b)) for a,b in zip(reference,pred))
                        equivalence.append({'workload':workload,'method':method,'batch':batch,
                            'different_positions_vs_fixed16':diff,'characters':sum(map(len,texts))})
                    print(workload,method,batch,repeat,round(elapsed,4),flush=True)
                except torch.OutOfMemoryError:
                    raw.append({'workload':workload,'method':method,'batch':batch,'repeat':repeat,'status':'OOM'})
                    gc.collect(); torch.cuda.empty_cache()
                write_jsonl(out/'raw.jsonl',raw)
        write_json(out/'equivalence.json',equivalence)
    summary=[]
    for workload in workloads:
        for method,batch in methods:
            subset=[r for r in raw if r['status']=='ok' and (r['workload'],r['method'],r['batch'])==(workload,method,batch)]
            if not subset: continue
            times=[r['elapsed_s'] for r in subset]
            summary.append({'workload':workload,'method':method,'batch':batch,'n':len(subset),
                'median_s':statistics.median(times),'min_s':min(times),'max_s':max(times),
                'p95_s':float(np.percentile(times,95)),
                'chars_per_s':sum(map(len,workloads[workload]))/statistics.median(times),
                'median_model_s':statistics.median(r['model_s'] for r in subset),
                'peak_allocated_mib':max(r['peak_allocated_mib'] for r in subset),
                'windows':subset[0]['windows'],'calls':subset[0]['forward_calls']})
    gains=[]
    for workload in workloads:
        selected_rows=[r for r in summary if r['workload']==workload]
        best=max((r for r in selected_rows if r['method']=='fast'),key=lambda r:r['chars_per_s'])
        sequential=next(r for r in selected_rows if r['method']=='sequential')
        fixed1=next(r for r in selected_rows if r['method']=='fixed' and r['batch']==1)
        fixed64=next(r for r in selected_rows if r['method']=='fixed' and r['batch']==64)
        fast64=next(r for r in selected_rows if r['method']=='fast' and r['batch']==64)
        gains.append({'workload':workload,'best_fast_batch':best['batch'],'total_speedup':sequential['median_s']/best['median_s'],
            'batch_only_fixed_1_to_64':fixed1['median_s']/fixed64['median_s'],
            'implementation_only_at_64':fixed64['median_s']/fast64['median_s'],
            'fast64_vs_sequential':sequential['median_s']/fast64['median_s']})
    write_json(out/'summary.json',summary)
    write_json(out/'gains.json',gains)
    write_json(out/'completed.json',{'status':'completed','configurations':len(summary),'measurements':len(raw),'repeats':repeats})


if __name__=='__main__':
    main()
