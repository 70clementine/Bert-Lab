"""Character-aligned BERT punctuation: preparation, training, inference and evaluation.

Each retained Unicode character occupies exactly one vocabulary position. Unknown
characters use UNK but the original character is retained for reconstruction.
No test text is used to select weights, thresholds or training hyperparameters.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
import unicodedata

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForTokenClassification, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
LABELS = ['O', '，', '。', '？', '！', '；', '：', '、']
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
PUNCT = {c: c for c in LABELS[1:]}
PUNCT.update({',': '，', '.': '。', '?': '？', '!': '！', ';': '；', ':': '：'})
END_IDS = {2, 3, 4}
IGNORE = -100


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def clean_label(text, stats=None):
    """Strip annotations and punctuation. A punctuation run uses its last target.

    Decimal ASCII/full-width dots between digits are retained as content. Quotes,
    brackets, ellipsis and dash are excluded from this task's label vocabulary.
    Spaces are corpus segmentation, not model input. The raw corpus is untouched.
    """
    stats = stats if stats is not None else Counter()
    stats['pinyin_annotations'] += len(re.findall(r'\{[^{}]*\}', text))
    text = re.sub(r'\{[^{}]*\}', '', text)
    text = re.sub(r'\s+', '', text)
    chars, ys = [], []
    for i, c in enumerate(text):
        decimal = c in '.．' and 0 < i < len(text)-1 and text[i-1].isdigit() and text[i+1].isdigit()
        if c in PUNCT and not decimal:
            stats['target_' + PUNCT[c]] += 1
            if chars:
                if ys[-1] > 0:
                    stats['collapsed_target_runs'] += 1
                ys[-1] = LABEL2ID[PUNCT[c]]
            else:
                stats['leading_target_dropped'] += 1
        elif unicodedata.category(c).startswith('P') and not decimal:
            stats['dropped_' + c] += 1
        elif unicodedata.category(c).startswith('C'):
            stats['control_dropped'] += 1
        else:
            chars.append(c)
            ys.append(0)
    return ''.join(chars), ys


class CharVocab:
    def __init__(self, directory):
        self.directory = Path(directory)
        # U+2028 is itself a vocabulary token. str.splitlines() splits it again
        # and silently shifts subsequent IDs away from the pretrained embeddings.
        # Match transformers.models.bert.tokenization_bert.load_vocab exactly.
        with (self.directory / 'vocab.txt').open(encoding='utf-8') as reader:
            self.tokens = [line.rstrip('\n') for line in reader.readlines()]
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        canonical_path = self.directory / 'tokenizer.json'
        if canonical_path.exists():
            canonical = read_json(canonical_path)['model']['vocab']
            if self.vocab != canonical:
                raise ValueError('Vocabulary IDs differ from the pretrained tokenizer.json')
        config_path = self.directory / 'config.json'
        if config_path.exists() and len(self.tokens) != read_json(config_path)['vocab_size']:
            raise ValueError('Vocabulary size differs from embedding table size')
        self.pad = self.vocab['[PAD]']
        self.cls = self.vocab['[CLS]']
        self.sep = self.vocab['[SEP]']
        self.unk = self.vocab['[UNK]']

    def encode(self, text):
        ids = []
        for c in text:
            # NFKC only when it maps to one character; output text is unchanged.
            normalized = unicodedata.normalize('NFKC', c).lower()
            token = normalized if len(normalized) == 1 else c
            ids.append(self.vocab.get(token, self.vocab.get(c, self.unk)))
        return ids


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    keep_start: int
    keep_end: int


def make_windows(n, window=400, stride=300):
    """Partition ownership first, then attach up to 50 characters of context.

    Ownership intervals are disjoint [k*stride, (k+1)*stride). First/last
    windows extend to document boundaries where possible. Every position has
    exactly one owner; no predicted punctuation controls future window starts.
    """
    if not (1 <= stride <= window <= 510):
        raise ValueError('Require 1 <= stride <= window <= 510 (CLS/SEP need 2).')
    if n == 0:
        return []
    if n <= window:
        return [Window(0, n, 0, n)]
    margin = (window - stride) // 2
    result = []
    for keep_start in range(0, n, stride):
        keep_end = min(n, keep_start + stride)
        start = max(0, keep_start - margin)
        end = min(n, start + window)
        if end == n:
            start = max(0, n - window)
        result.append(Window(start, end, keep_start, keep_end))
    return result


def prepare(cfg):
    vocab = CharVocab(ROOT / cfg['model_dir'])
    source = next((ROOT / '1998语料').glob('*1998-01*.txt'))
    raw = source.read_bytes()
    lines = raw.decode('gb18030').splitlines()
    grouped = defaultdict(list)
    stats = Counter()
    for line in lines:
        match = re.match(r'^(\d{8}-\d+-\d+)-\d+\s+(.*)$', line)
        if not match:
            if line.strip():
                stats['unparsed_nonempty_lines'] += 1
            continue
        text, labels = clean_label(match[2], stats)
        if not text:
            continue
        # Missing punctuation at a paragraph/title end is ambiguous, not O.
        if labels[-1] == 0:
            labels[-1] = IGNORE
            stats['unknown_paragraph_ends'] += 1
        grouped[match[1]].append((text, labels))
    records, seen = [], set()
    for docid, paragraphs in sorted(grouped.items()):
        text = ''.join(p[0] for p in paragraphs)
        ys = [y for p in paragraphs for y in p[1]]
        if len(text) < 20 or not any(y > 0 for y in ys):
            stats['excluded_short_or_unlabelled_articles'] += 1
            continue
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in seen:
            stats['duplicate_articles_removed'] += 1
            continue
        seen.add(h)
        records.append({'id': docid, 'text': text, 'labels': ys, 'hash': h,
                        'paragraph_hashes': [hashlib.sha256(p[0].encode()).hexdigest() for p in paragraphs if len(p[0]) >= 40]})
    # Articles sharing a >=40-char paragraph stay in the same split (union-find).
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, row in enumerate(records):
        for h in row['paragraph_hashes']:
            if h in owners:
                parent[find(i)] = find(owners[h])
            else:
                owners[h] = i
    clusters = defaultdict(list)
    for i, row in enumerate(records):
        clusters[find(i)].append(row)
    groups = list(clusters.values())
    random.Random(cfg['seed']).shuffle(groups)
    n = len(groups)
    cut1, cut2 = int(n*.8), int(n*.9)
    splits = {name: [r for g in subset for r in g] for name, subset in
              [('train', groups[:cut1]), ('val', groups[cut1:cut2]), ('test', groups[cut2:])]}
    manifest = {'source': str(source.relative_to(ROOT)), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'encoding': 'gb18030', 'seed': cfg['seed'], 'source_lines': len(lines),
                'cleaning': dict(stats), 'dedup_clusters': n, 'splits': {}, 'labels': LABELS}
    for name, rows in splits.items():
        for row in rows:
            row['input_ids'] = vocab.encode(row['text'])
        write_jsonl(ROOT / f'data/processed/{name}.jsonl', rows)
        count = Counter(y for r in rows for y in r['labels'])
        manifest['splits'][name] = {'articles': len(rows), 'characters': sum(len(r['text']) for r in rows),
            'windows': sum(len(make_windows(len(r['text']), cfg['window'], cfg['stride'])) for r in rows),
            'label_counts': {str(k): v for k, v in sorted(count.items())},
            'unknown_tokens': sum(r['input_ids'].count(vocab.unk) for r in rows),
            'ids': [r['id'] for r in rows]}
    checks = {}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        ah = {h for r in splits[a] for h in r['paragraph_hashes']}
        bh = {h for r in splits[b] for h in r['paragraph_hashes']}
        checks[a+'_'+b] = {'article_overlap': len({r['hash'] for r in splits[a]} & {r['hash'] for r in splits[b]}),
                             'paragraph_40plus_overlap': len(ah & bh)}
        assert not checks[a+'_'+b]['article_overlap'] and not checks[a+'_'+b]['paragraph_40plus_overlap']
    manifest['leakage_checks'] = checks
    write_json(ROOT / 'data/processed/manifest.json', manifest)
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'ids'} for k, v in manifest['splits'].items()}, ensure_ascii=False, indent=2), flush=True)


class WindowDataset(Dataset):
    def __init__(self, rows, cfg):
        self.rows = rows
        self.items = [(i, w) for i, r in enumerate(rows)
                      for w in make_windows(len(r['text']), cfg['window'], cfg['stride'])
                      if any(y != IGNORE for y in r['labels'][w.keep_start:w.keep_end])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        doc, w = self.items[i]
        row = self.rows[doc]
        labels = [y if w.keep_start <= k < w.keep_end else IGNORE
                  for k, y in enumerate(row['labels'][w.start:w.end], start=w.start)]
        return row['input_ids'][w.start:w.end], labels


def collate(examples, vocab, device='cpu'):
    length = max(len(x[0]) for x in examples) + 2
    # Pad to 8 for tensor-core-friendly shapes, always <= 512.
    length = math.ceil(length / 8) * 8
    ids = torch.full((len(examples), length), vocab.pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, IGNORE)
    for i, (x, y) in enumerate(examples):
        ids[i, :len(x)+2] = torch.tensor([vocab.cls] + x + [vocab.sep])
        mask[i, :len(x)+2] = 1
        if y is not None:
            labels[i, 1:len(x)+1] = torch.tensor(y)
    return {'input_ids': ids.to(device), 'attention_mask': mask.to(device), 'labels': labels.to(device)}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure(cfg):
    torch.set_num_threads(cfg['cpu_threads'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the recorded experiment. No silent CPU fallback.')
    if cfg['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This configuration requires BF16 support.')
    seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True


def amp(cfg):
    return torch.autocast('cuda', dtype=torch.bfloat16) if cfg['precision'] == 'bf16' else nullcontext()


def cm_update(cm, gold, pred):
    gold, pred = np.asarray(gold).reshape(-1), np.asarray(pred).reshape(-1)
    ok = gold != IGNORE
    cm += np.bincount(gold[ok]*len(LABELS)+pred[ok], minlength=len(LABELS)**2).reshape(len(LABELS), len(LABELS))


def metrics(cm):
    per = {}
    for i, label in enumerate(LABELS):
        tp, support, predicted = int(cm[i, i]), int(cm[i].sum()), int(cm[:, i].sum())
        precision, recall = tp/max(1, predicted), tp/max(1, support)
        per[label] = {'precision': precision, 'recall': recall, 'f1': 2*tp/max(1, support+predicted), 'support': support, 'predicted': predicted}
    tp = int(np.trace(cm[1:, 1:]))
    gold, pred = int(cm[1:].sum()), int(cm[:, 1:].sum())
    # Wrong punctuation type contributes one FP and one FN.
    detected = int(cm[1:, 1:].sum())
    end = sorted(END_IDS)
    end_tp = int(cm[np.ix_(end, end)].sum())
    end_gold, end_pred = int(cm[end, :].sum()), int(cm[:, end].sum())
    return {'punctuation_micro_precision': tp/max(1, pred), 'punctuation_micro_recall': tp/max(1, gold),
            'punctuation_micro_f1': 2*tp/max(1, gold+pred),
            'punctuation_macro_f1': float(np.mean([per[l]['f1'] for l in LABELS[1:]])),
            'boundary_f1': 2*detected/max(1, gold+pred),
            'sentence_end_f1': 2*end_tp/max(1, end_gold+end_pred),
            'all_position_accuracy': float(np.trace(cm)/max(1, cm.sum())),
            'evaluated_positions': int(cm.sum()), 'per_label': per, 'confusion_matrix': cm.tolist()}


@torch.inference_mode()
def evaluate_loader(model, loader, cfg):
    model.eval()
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    loss_sum, count = 0.0, 0
    for batch in loader:
        batch = {k: v.cuda() for k, v in batch.items()}
        with amp(cfg):
            out = model(**batch)
        n = int((batch['labels'] != IGNORE).sum())
        loss_sum += out.loss.item()*n
        count += n
        cm_update(cm, batch['labels'].cpu().numpy(), out.logits.argmax(-1).cpu().numpy())
    result = metrics(cm)
    result['loss'] = loss_sum/max(1, count)
    return result


def train(cfg, pilot=False):
    configure(cfg)
    run = ROOT / ('runs/pilot' if pilot else cfg.get('run_dir', 'runs/main'))
    if not pilot and (run/'completed.json').exists():
        raise FileExistsError(f'Completed experiment already exists: {run}. Use a new run_dir or archive it before retraining.')
    vocab = CharVocab(ROOT / cfg['model_dir'])
    data_dir = ROOT / cfg.get('data_dir', 'data/processed')
    train_rows = jsonl(data_dir / 'train.jsonl')
    val_rows = jsonl(data_dir / 'val.jsonl')
    for row in train_rows + val_rows:
        if row['input_ids'] != vocab.encode(row['text']):
            raise ValueError(f"Stale or incompatible token-ID cache in article {row['id']}; regenerate the data.")
    ds = WindowDataset(train_rows, cfg)
    from functools import partial
    loader = DataLoader(ds, batch_size=cfg['train_batch_size'], shuffle=True, num_workers=0,
                        collate_fn=partial(collate, vocab=vocab), generator=torch.Generator().manual_seed(cfg['seed']))
    val_loader = DataLoader(WindowDataset(val_rows, cfg), batch_size=cfg['eval_batch_size'], shuffle=False,
                            num_workers=0, collate_fn=partial(collate, vocab=vocab))
    initial_checkpoint = ROOT / cfg.get('initial_checkpoint', cfg['model_dir'])
    model = BertForTokenClassification.from_pretrained(initial_checkpoint, num_labels=len(LABELS),
                id2label=dict(enumerate(LABELS)), label2id=LABEL2ID, attn_implementation='sdpa').cuda()
    model.config.char_vocab_reader = 'lf_v2'
    class_weights = None
    if cfg.get('class_weighting') == 'sqrt_inverse_capped':
        counts = Counter(y for r in train_rows for y in r['labels'] if y != IGNORE)
        weights = [min(cfg.get('class_weight_cap', 8.0), math.sqrt(counts[0]/max(1, counts[i]))) for i in range(len(LABELS))]
        class_weights = torch.tensor(weights, dtype=torch.float32, device='cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    accum = cfg['gradient_accumulation']
    updates_per_epoch = math.ceil(len(loader)/accum)
    total_steps = updates_per_epoch * cfg['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*cfg['warmup_ratio']), total_steps)
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / 'config.json', cfg)
    write_json(run / 'class_weights.json', {'method': cfg.get('class_weighting', 'none'),
        'weights': class_weights.cpu().tolist() if class_weights is not None else [1.0]*len(LABELS),
        'labels': LABELS, 'source': 'training split only'})
    write_json(run / 'environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
        'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
    start = time.perf_counter()
    history, best, update, examples = [], -1.0, 0, 0
    torch.cuda.reset_peak_memory_stats()
    log = (run / 'steps.jsonl').open('w', encoding='utf-8')
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(1, cfg['epochs']+1):
            model.train()
            epoch_start = time.perf_counter()
            loss_sum, objective_sum, positions = 0.0, 0.0, 0
            for batch_index, batch in enumerate(loader):
                batch = {k: v.cuda() for k, v in batch.items()}
                n = int((batch['labels'] != IGNORE).sum())
                group_size = min(accum, len(loader) - (batch_index//accum)*accum)
                with amp(cfg):
                    out = model(**batch)
                    objective = out.loss if class_weights is None else torch.nn.functional.cross_entropy(
                        out.logits.float().reshape(-1, len(LABELS)), batch['labels'].reshape(-1),
                        weight=class_weights, ignore_index=IGNORE)
                    loss = objective/group_size
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss; refusing to continue.')
                loss.backward()
                loss_sum += out.loss.item()*n
                objective_sum += objective.item()*n
                positions += n
                examples += batch['input_ids'].shape[0]
                if (batch_index+1) % accum == 0 or batch_index+1 == len(loader):
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    update += 1
                    if update % 20 == 0 or (pilot and update >= 20):
                        torch.cuda.synchronize()
                        row = {'epoch': epoch, 'update': update, 'batch': batch_index+1,
                            'loss': loss_sum/max(1, positions), 'lr': scheduler.get_last_lr()[0],
                            'grad_norm': float(norm), 'elapsed_s': time.perf_counter()-start,
                            'examples': examples, 'peak_memory_mib': torch.cuda.max_memory_allocated()/1024**2}
                        log.write(json.dumps(row)+'\n')
                        log.flush()
                        print(json.dumps(row), flush=True)
                    if pilot and update >= 20:
                        write_json(run / 'result.json', {'elapsed_s': time.perf_counter()-start, 'examples': examples,
                            'train_windows': len(ds), 'batches_per_epoch': len(loader), 'updates_per_epoch': updates_per_epoch,
                            'peak_memory_mib': torch.cuda.max_memory_allocated()/1024**2})
                        return
            torch.cuda.synchronize()
            training_s = time.perf_counter()-epoch_start
            validation = evaluate_loader(model, val_loader, cfg)
            record = {'epoch': epoch, 'train_loss': loss_sum/max(1, positions), 'train_s': training_s,
                      'train_objective_loss': objective_sum/max(1, positions),
                      'validation': validation, 'elapsed_s': time.perf_counter()-start}
            history.append(record)
            print('EPOCH', json.dumps(record, ensure_ascii=False), flush=True)
            if validation['punctuation_micro_f1'] > best:
                best = validation['punctuation_micro_f1']
                ckpt = run / 'best'
                model.save_pretrained(ckpt)
                (ckpt / 'vocab.txt').write_bytes((ROOT / cfg['model_dir'] / 'vocab.txt').read_bytes())
                write_json(ckpt / 'selection.json', {'epoch': epoch, 'validation_micro_f1': best, 'criterion': 'max validation punctuation micro F1 (excluding O)'})
            write_json(run / 'history.json', history)
        write_json(run / 'completed.json', {'status': 'completed', 'epochs': cfg['epochs'], 'optimizer_updates': update,
            'elapsed_s': time.perf_counter()-start, 'best_validation_micro_f1': best,
            'peak_memory_mib': torch.cuda.max_memory_allocated()/1024**2,
            'finished_at_utc': datetime.now(timezone.utc).isoformat()})
    finally:
        log.close()


class Predictor:
    def __init__(self, cfg, checkpoint=None):
        configure(cfg)
        self.cfg = cfg
        self.directory = Path(checkpoint or ROOT / cfg.get('checkpoint', 'runs/main/best'))
        self.vocab = CharVocab(self.directory)
        self.model = BertForTokenClassification.from_pretrained(self.directory, attn_implementation='sdpa').cuda().eval()
        if getattr(self.model.config, 'char_vocab_reader', None) != 'lf_v2':
            raise ValueError('This checkpoint predates the vocabulary-ID fix. Use a retrained v2 checkpoint; legacy weights require the archived legacy encoder.')
        self.last_stats = {}

    @torch.inference_mode()
    def forward(self, segments):
        batch = collate([(s, None) for s in segments], self.vocab, 'cuda')
        batch.pop('labels')
        with amp(self.cfg):
            logits = self.model(**batch).logits
        return logits

    def fixed(self, texts, batch_size=16, fusion=False):
        """Batch across windows and documents; scatter using document/position IDs."""
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        encoded = [self.vocab.encode(t) for t in texts]
        jobs = [(d, w) for d, text in enumerate(texts) for w in make_windows(len(text), self.cfg['window'], self.cfg['stride'])]
        # Bucket by length, retaining source indices for exact output restoration.
        jobs.sort(key=lambda item: item[1].end-item[1].start, reverse=True)
        result = [np.zeros(len(t), dtype=np.int64) for t in texts]
        coverage = [np.zeros(len(t), dtype=np.int16) for t in texts]
        scores = [np.zeros((len(t), len(LABELS)), dtype=np.float32) for t in texts] if fusion else None
        model_s, calls, padded_tokens, real_tokens = 0.0, 0, 0, 0
        for offset in range(0, len(jobs), batch_size):
            group = jobs[offset:offset+batch_size]
            segments = [encoded[d][w.start:w.end] for d, w in group]
            # Separate kernel timing from host preprocessing / transfers.
            batch = collate([(s, None) for s in segments], self.vocab, 'cuda')
            batch.pop('labels')
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            with torch.inference_mode(), amp(self.cfg):
                logits = self.model(**batch).logits
            end.record()
            end.synchronize()
            model_s += begin.elapsed_time(end)/1000
            values = logits.float().softmax(-1).cpu().numpy() if fusion else logits.argmax(-1).cpu().numpy()
            calls += 1
            padded_tokens += batch['input_ids'].numel()
            real_tokens += sum(len(s)+2 for s in segments)
            for j, (d, w) in enumerate(group):
                if fusion:
                    size = w.end-w.start
                    pos = np.arange(size)
                    weight = np.minimum(np.minimum(pos+1, size-pos), 51).astype(np.float32)/51
                    scores[d][w.start:w.end] += values[j, 1:size+1] * weight[:, None]
                    coverage[d][w.start:w.end] += 1
                else:
                    result[d][w.keep_start:w.keep_end] = values[j, 1+w.keep_start-w.start:1+w.keep_end-w.start]
                    coverage[d][w.keep_start:w.keep_end] += 1
        for d in range(len(texts)):
            assert np.all(coverage[d] >= 1) if fusion else np.all(coverage[d] == 1)
            if fusion:
                result[d] = scores[d].argmax(-1)
        self.last_stats = {'windows': len(jobs), 'forward_calls': calls, 'model_s': model_s,
                           'padded_tokens': padded_tokens, 'real_tokens': real_tokens}
        return result

    def sequential(self, texts):
        """Assignment baseline: <=400 chars, keep <=300, end at last predicted 。.

        If no period is predicted before the stride, advance by stride anyway;
        otherwise an unpunctuated passage could stall forever.
        """
        result, windows, model_s = [], 0, 0.0
        for text in texts:
            ids = self.vocab.encode(text)
            output, start = [], 0
            while start < len(ids):
                segment = ids[start:start+self.cfg['window']]
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                batch = collate([(segment, None)], self.vocab, 'cuda')
                batch.pop('labels')
                begin.record()
                with torch.inference_mode(), amp(self.cfg):
                    logits = self.model(**batch).logits
                end.record()
                end.synchronize()
                model_s += begin.elapsed_time(end)/1000
                pred = logits.argmax(-1)[0, 1:len(segment)+1].cpu().tolist()
                if start+len(segment) == len(ids):
                    take = len(segment)
                else:
                    limit = min(self.cfg['stride'], len(segment))
                    stops = [i+1 for i, p in enumerate(pred[:limit]) if p == LABEL2ID['。']]
                    take = stops[-1] if stops else limit
                output.extend(pred[:take])
                start += take
                windows += 1
            assert len(output) == len(text)
            result.append(np.asarray(output, dtype=np.int64))
        self.last_stats = {'windows': windows, 'forward_calls': windows, 'model_s': model_s}
        return result

    @torch.inference_mode()
    def fixed_fast(self, texts, batch_size=16):
        """Same ownership/output as fixed(), but one final device-to-host copy.

        Avoid allocating/transferring unused training-label tensors. Enqueue GPU
        batches without per-batch synchronization, collect compact owner slices,
        then copy all labels once. GPU label storage grows linearly with N (8N
        bytes plus window padding and concatenation buffers), so this is intended
        for finite documents, not an unbounded stream.
        """
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        encoded=[self.vocab.encode(t) for t in texts]
        jobs=[(d,w) for d,t in enumerate(texts)
              for w in make_windows(len(t),self.cfg['window'],self.cfg['stride'])]
        jobs.sort(key=lambda item:item[1].end-item[1].start,reverse=True)
        result=[np.zeros(len(t),dtype=np.int64) for t in texts]
        if not jobs:
            self.last_stats={'windows':0,'forward_calls':0,'model_s':0.0,'padded_tokens':0,'real_tokens':0}
            return result
        slices,events=[],[]
        real_tokens=padded_tokens=calls=0
        with amp(self.cfg):
            for offset in range(0,len(jobs),batch_size):
                group=jobs[offset:offset+batch_size]
                width=math.ceil((max(w.end-w.start for _,w in group)+2)/8)*8
                ids=torch.full((len(group),width),self.vocab.pad,dtype=torch.long,pin_memory=True)
                mask=torch.zeros((len(group),width),dtype=torch.long,pin_memory=True)
                for i,(d,w) in enumerate(group):
                    segment=encoded[d][w.start:w.end]
                    ids[i,:len(segment)+2]=torch.tensor([self.vocab.cls]+segment+[self.vocab.sep])
                    mask[i,:len(segment)+2]=1
                    real_tokens+=len(segment)+2
                gpu_ids=ids.cuda(non_blocking=True)
                gpu_mask=mask.cuda(non_blocking=True)
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                logits=self.model(input_ids=gpu_ids,attention_mask=gpu_mask).logits
                end.record()
                events.append((begin,end))
                pred=logits.argmax(-1)
                for i,(_,w) in enumerate(group):
                    slices.append(pred[i,1+w.keep_start-w.start:1+w.keep_end-w.start])
                padded_tokens+=ids.numel()
                calls+=1
        combined=torch.cat(slices).cpu().numpy()
        cursor=0
        for d,w in jobs:
            size=w.keep_end-w.keep_start
            result[d][w.keep_start:w.keep_end]=combined[cursor:cursor+size]
            cursor+=size
        assert cursor==sum(map(len,texts))
        self.last_stats={'windows':len(jobs),'forward_calls':calls,
            'model_s':sum(a.elapsed_time(b) for a,b in events)/1000,
            'padded_tokens':padded_tokens,'real_tokens':real_tokens}
        return result


def render(text, pred):
    if len(text) != len(pred):
        raise ValueError('Character/label alignment mismatch')
    return ''.join(c + (LABELS[int(p)] if p > 0 else '') for c, p in zip(text, pred))


def evaluate_predictions(rows, predictions, cfg):
    assert len(rows) == len(predictions)
    overall = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    boundary = np.zeros_like(overall)
    interior = np.zeros_like(overall)
    long_cm = np.zeros_like(overall)
    for row, pred in zip(rows, predictions):
        assert len(row['labels']) == len(pred)
        ys = np.asarray(row['labels'])
        cm_update(overall, ys, pred)
        near = np.zeros(len(ys), dtype=bool)
        windows = make_windows(len(ys), cfg['window'], cfg['stride'])
        for w in windows[1:]:
            near[max(0,w.keep_start-10):min(len(ys),w.keep_start+10)] = True
        cm_update(boundary, ys[near], pred[near])
        cm_update(interior, ys[~near], pred[~near])
        if len(ys) > 512:
            cm_update(long_cm, ys, pred)
    return {'overall': metrics(overall), 'seam_plus_minus_10': metrics(boundary),
            'non_seam': metrics(interior), 'articles_over_512_chars': metrics(long_cm)}


def test(cfg):
    predictor = Predictor(cfg)
    rows = jsonl(ROOT / 'data/processed/test.jsonl')
    texts = [r['text'] for r in rows]
    outdir = ROOT / 'runs/main/test'
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    for method in ['all_O', 'fixed', 'sequential', 'fusion']:
        start = time.perf_counter()
        if method == 'all_O':
            predictions = [np.zeros(len(t), dtype=np.int64) for t in texts]
        elif method == 'sequential':
            predictions = predictor.sequential(texts)
        else:
            predictions = predictor.fixed(texts, cfg['eval_batch_size'], fusion=method=='fusion')
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        result = evaluate_predictions(rows, predictions, cfg)
        result['elapsed_s'] = elapsed
        results[method] = result
        write_json(outdir / 'metrics.json', results)
        write_jsonl(outdir / f'{method}_predictions.jsonl',
            [{'id': r['id'], 'prediction': p.tolist()} for r, p in zip(rows, predictions)])
        print('TEST', method, result['overall']['punctuation_micro_f1'], 'seconds', elapsed, flush=True)
    fixed = jsonl(outdir / 'fixed_predictions.jsonl')
    errors = []
    for row, output in zip(rows, fixed):
        gold, pred = np.asarray(row['labels']), np.asarray(output['prediction'])
        mismatches = np.flatnonzero((gold != IGNORE) & (gold != pred))
        for position in mismatches[:3]:
            lo, hi = max(0, position-35), min(len(gold), position+36)
            errors.append({'article_id': row['id'], 'position': int(position), 'character': row['text'][position],
                'gold_label': LABELS[gold[position]], 'predicted_label': LABELS[pred[position]],
                'input': row['text'][lo:hi], 'gold_context': render(row['text'][lo:hi], gold[lo:hi]),
                'predicted_context': render(row['text'][lo:hi], pred[lo:hi])})
    write_jsonl(outdir / 'errors.jsonl', errors)
    # Public-domain excerpts: illustrative transfer examples, NOT a held-out gold benchmark.
    examples = [
        {'title': '《论语·学而》节选', 'text': '学而时习之不亦说乎有朋自远方来不亦乐乎人不知而不愠不亦君子乎', 'reference': '学而时习之，不亦说乎？有朋自远方来，不亦乐乎？人不知而不愠，不亦君子乎？'},
        {'title': '《桃花源记》节选', 'text': '晋太元中武陵人捕鱼为业缘溪行忘路之远近忽逢桃花林夹岸数百步中无杂树芳草鲜美落英缤纷渔人甚异之复前行欲穷其林', 'reference': '晋太元中，武陵人捕鱼为业。缘溪行，忘路之远近。忽逢桃花林，夹岸数百步，中无杂树，芳草鲜美，落英缤纷。渔人甚异之，复前行，欲穷其林。'},
        {'title': '《岳阳楼记》节选', 'text': '嗟夫予尝求古仁人之心或异二者之为何哉不以物喜不以己悲居庙堂之高则忧其民处江湖之远则忧其君', 'reference': None}]
    demo_preds = predictor.fixed([r['text'] for r in examples], cfg['eval_batch_size'])
    for r, p in zip(examples, demo_preds):
        r['prediction'] = render(r['text'], p)
    write_json(outdir / 'classical_examples.json', examples)
    write_json(outdir / 'completed.json', {'status': 'completed', 'articles': len(rows), 'methods': list(results),
        'checkpoint_selection': read_json(ROOT / 'runs/main/best/selection.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'pilot', 'train', 'test', 'predict'])
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--input', help='UTF-8 text file for prediction')
    parser.add_argument('--output', help='Output UTF-8 text file')
    parser.add_argument('--text')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--method', choices=['fixed', 'fast', 'fusion', 'sequential'], default='fast')
    args = parser.parse_args()
    cfg = read_json(args.config)
    if args.command == 'prepare':
        if cfg.get('data_dir') == 'data/v2':
            from revision_data import main as prepare_v2
            prepare_v2()
        else:
            prepare(cfg)
    elif args.command in ['pilot', 'train']:
        train(cfg, pilot=args.command=='pilot')
    elif args.command == 'test':
        if cfg.get('data_dir') == 'data/v2':
            from revision_evaluate import main as evaluate_v2
            evaluate_v2()
        else:
            raise ValueError('Legacy tests are archived. Use the v2 config and revision_evaluate.py.')
    else:
        raw = Path(args.input).read_text(encoding='utf-8-sig') if args.input else args.text
        if raw is None:
            parser.error('predict requires --text or --input')
        text, _ = clean_label(raw)
        predictor = Predictor(cfg)
        if args.method=='sequential':
            pred=predictor.sequential([text])[0]
        elif args.method=='fast':
            pred=predictor.fixed_fast([text],args.batch_size)[0]
        else:
            pred=predictor.fixed([text],args.batch_size,fusion=args.method=='fusion')[0]
        output = render(text, pred)
        if args.output:
            Path(args.output).write_text(output, encoding='utf-8')
        else:
            print(output)


if __name__ == '__main__':
    main()
