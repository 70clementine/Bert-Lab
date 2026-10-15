"""Freeze the v1 split/labels and correct IDs; prepare unseen February holdout."""
from collections import Counter, defaultdict
import hashlib
import re
from bert_punct import ROOT, CharVocab, IGNORE, clean_label, jsonl, read_json, write_json, write_jsonl


def main():
    # A fresh clone has no cached January split. Build it deterministically from
    # the user-provided source before applying the shared evaluation preparation.
    if not (ROOT/'data/processed/manifest.json').exists():
        from bert_punct import prepare
        prepare(read_json(ROOT/'config.json'))
    vocab = CharVocab(ROOT/'models/bert-base-chinese')
    canonical = read_json(ROOT/'models/bert-base-chinese/tokenizer.json')['model']['vocab']
    original_text = (ROOT/'models/bert-base-chinese/vocab.txt').read_text(encoding='utf-8')
    old_vocab = {t:i for i,t in enumerate(original_text.splitlines())}
    mapping = {'cause':'str.splitlines() treats U+2028 (and embedded line separators) as line breaks',
               'old_entries':len(original_text.splitlines()), 'canonical_entries':len(canonical),
               'mismatched_token_ids':sum(old_vocab.get(t)!=i for t,i in canonical.items()),
               'first_special_case':{'token':'U+2028','canonical_id':343,'old_id':old_vocab.get('\u2028')},
               'example':{'token':'我','old_id':old_vocab['我'],'canonical_id':canonical['我']},
               'fixed_dictionary_matches_canonical':vocab.vocab==canonical}
    write_json(ROOT/'artifacts/v2/vocab_diagnosis.json',mapping)
    manifest = read_json(ROOT/'data/processed/manifest.json')
    manifest['encoding_revision']='lf_v2'
    manifest['split_policy']='exactly the original January article split; text/labels unchanged; only token IDs corrected'
    all_january=[]
    for name in ['train','val','test']:
        rows=jsonl(ROOT/f'data/processed/{name}.jsonl')
        changed=total=0
        for r in rows:
            corrected=vocab.encode(r['text'])
            changed+=sum(a!=b for a,b in zip(r['input_ids'],corrected))
            total+=len(corrected)
            r['input_ids']=corrected
        write_jsonl(ROOT/f'data/v2/{name}.jsonl',rows)
        manifest['splits'][name]['corrected_id_positions']=changed
        manifest['splits'][name]['unknown_tokens']=sum(r['input_ids'].count(vocab.unk) for r in rows)
        all_january.extend(rows)
    write_json(ROOT/'data/v2/manifest.json',manifest)
    # Entire February is reserved for final evaluation. No training, tuning or
    # checkpoint selection uses its labels or model predictions.
    source=next((ROOT/'1998语料').glob('*1998-02*.txt'))
    raw=source.read_bytes()
    grouped=defaultdict(list)
    stats=Counter()
    for line in raw.decode('gb18030').splitlines():
        m=re.match(r'^(\d{8}-\d+-\d+)-\d+\s+(.*)$',line)
        if not m:
            if line.strip(): stats['unparsed_nonempty_lines']+=1
            continue
        text,ys=clean_label(m[2],stats)
        if not text: continue
        if ys[-1]==0: ys[-1]=IGNORE
        grouped[m[1]].append((text,ys))
    january_hashes={r['hash'] for r in all_january}
    january_paragraphs={h for r in all_january for h in r['paragraph_hashes']}
    seen=set()
    rows=[]
    for docid, paragraphs in sorted(grouped.items()):
        text=''.join(p[0] for p in paragraphs)
        ys=[y for p in paragraphs for y in p[1]]
        if len(text)<20 or not any(y>0 for y in ys):
            stats['excluded_short_or_unlabelled']+=1
            continue
        h=hashlib.sha256(text.encode()).hexdigest()
        ph=[hashlib.sha256(p[0].encode()).hexdigest() for p in paragraphs if len(p[0])>=40]
        if h in january_hashes or set(ph)&january_paragraphs:
            stats['excluded_january_exact_overlap']+=1
            continue
        if h in seen:
            stats['excluded_february_duplicate']+=1
            continue
        seen.add(h)
        rows.append({'id':docid,'text':text,'labels':ys,'input_ids':vocab.encode(text),'hash':h,'paragraph_hashes':ph})
    write_jsonl(ROOT/'data/v2/february_holdout.jsonl',rows)
    write_json(ROOT/'data/v2/february_manifest.json',{'role':'frozen final out-of-month test; not for tuning',
        'source':str(source.relative_to(ROOT)), 'source_sha256':hashlib.sha256(raw).hexdigest(),
        'articles':len(rows), 'characters':sum(len(r['text']) for r in rows), 'cleaning':dict(stats),
        'january_exact_article_overlap':0,'january_exact_paragraph_40plus_overlap':0,
        'ids':[r['id'] for r in rows]})
    print(mapping)
    print('Frozen February holdout',len(rows),'articles;',sum(len(r['text']) for r in rows),'characters')


if __name__=='__main__':
    main()
