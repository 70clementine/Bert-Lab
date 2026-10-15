"""Freeze a public, editorially punctuated Analects reference set (no training)."""
from collections import Counter
import hashlib
import json
import urllib.request
from urllib.parse import quote
from bert_punct import ROOT, CharVocab, IGNORE, clean_label, write_json, write_jsonl

REVISION='b8594f81a89752241442f2ce267d6f66f96704ee'


def main():
    base=f'https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/{REVISION}/'
    url=base+quote('论语/lunyu.json')
    folder=ROOT/'data/external'
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/'lunyu.json'
    if not target.exists():
        with urllib.request.urlopen(url,timeout=60) as response:
            target.write_bytes(response.read())
    raw=target.read_bytes()
    chapters=json.loads(raw)
    rows=[]
    stats=Counter()
    vocab=CharVocab(ROOT/'models/bert-base-chinese')
    for i,chapter in enumerate(chapters):
        parts=[]
        for paragraph in chapter['paragraphs']:
            text,ys=clean_label(paragraph,stats)
            if not text: continue
            if ys[-1]==0: ys[-1]=IGNORE
            parts.append((text,ys))
        text=''.join(x[0] for x in parts)
        ys=[y for p in parts for y in p[1]]
        rows.append({'id':f'lunyu-{i+1:02d}','title':chapter['chapter'],'text':text,'labels':ys,
                     'input_ids':vocab.encode(text),'hash':hashlib.sha256(text.encode()).hexdigest()})
    assert len(rows)==20
    write_jsonl(ROOT/'data/v2/classical_reference.jsonl',rows)
    write_json(ROOT/'data/v2/classical_manifest.json',{'role':'external classical reference; all 20 chapters; no tuning',
        'url':url,'repo':'https://github.com/chinese-poetry/chinese-poetry','commit':REVISION,
        'sha256':hashlib.sha256(raw).hexdigest(),'chapters':len(rows),'characters':sum(len(r['text']) for r in rows),
        'paragraphs':sum(len(c['paragraphs']) for c in chapters),'normalization':dict(stats),
        'limitations':['One famous work; not representative of all classical Chinese',
                      'Public text may occur in pretrained BERT data; no contamination-free claim',
                      'Punctuation is this digital edition, not a unique historical truth',
                      'Seven target punctuation types; quotes and brackets excluded']})
    print('Frozen classical reference:',len(rows),'chapters',sum(len(r['text']) for r in rows),'characters')


if __name__=='__main__':
    main()
