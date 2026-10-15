"""Focused tests for leakage-prone preprocessing, alignment, windowing and scoring."""
import unittest
import numpy as np
from bert_punct import (CharVocab, IGNORE, LABEL2ID, ROOT, WindowDataset, clean_label, cm_update,
                        collate, make_windows, metrics, render)


class CorrectnessTests(unittest.TestCase):
    def test_entire_vocabulary_matches_official_tokenizer(self):
        import json
        directory = ROOT / 'models/bert-base-chinese'
        canonical = json.loads((directory/'tokenizer.json').read_text(encoding='utf-8'))['model']['vocab']
        vocab = CharVocab(directory)
        self.assertEqual(len(vocab.tokens), 21128)
        self.assertEqual(vocab.vocab, canonical)
        self.assertEqual(vocab.vocab['\u2028'], 343)
        self.assertEqual(vocab.vocab['我'], 2769)

    def test_chinese_character_ids_match_standard_tokenizer(self):
        from transformers import BertTokenizerFast
        directory = ROOT / 'models/bert-base-chinese'
        vocab = CharVocab(directory)
        tokenizer = BertTokenizerFast.from_pretrained(directory)
        text = '我们一定能够克服这些困难继续稳步前进学而时习之不亦说乎'
        self.assertEqual(vocab.encode(text), tokenizer(text,add_special_tokens=False)['input_ids'])

    def test_annotations_quotes_decimal_and_fullwidth(self):
        text, labels = clean_label('上{shang5} 海 “ 增长 ３．５％ ” ， 好 吗 ？')
        self.assertEqual(text, '上海增长３．５好吗')
        self.assertEqual(labels[text.index('５')], LABEL2ID['，'])
        self.assertEqual(labels[-1], LABEL2ID['？'])
        self.assertNotIn('shang', text)

    def test_punctuation_run_and_empty(self):
        self.assertEqual(clean_label('？！'), ('', []))
        self.assertEqual(clean_label('好？！'), ('好', [LABEL2ID['！']]))
        self.assertEqual(clean_label(''), ('', []))

    def test_every_character_owned_once_and_context_contains_owner(self):
        for n in list(range(0, 850)) + [999, 1000, 3000, 10000, 100001]:
            cover = np.zeros(n, dtype=np.int16)
            for w in make_windows(n):
                self.assertTrue(0 <= w.start <= w.keep_start < w.keep_end <= w.end <= n)
                self.assertLessEqual(w.end-w.start, 400)
                cover[w.keep_start:w.keep_end] += 1
            self.assertTrue(np.all(cover == 1), n)

    def test_invalid_window_parameters(self):
        for win, stride in [(511, 300), (400, 0), (300, 400)]:
            with self.assertRaises(ValueError):
                make_windows(1000, win, stride)

    def test_training_skips_windows_with_no_supervision(self):
        row = {'text': '甲'*601, 'input_ids': [1]*601, 'labels': [0]*600+[IGNORE]}
        ds = WindowDataset([row], {'window': 400, 'stride': 300})
        self.assertEqual(len(ds), 2)
        self.assertEqual(sum(sum(y != IGNORE for y in ds[i][1]) for i in range(len(ds))), 600)

    def test_metrics_wrong_type_is_both_fp_and_fn(self):
        cm = np.zeros((8, 8), dtype=np.int64)
        cm_update(cm, [0, 1, 2, IGNORE], [0, 1, 1, 7])
        m = metrics(cm)
        self.assertAlmostEqual(m['punctuation_micro_f1'], .5)
        self.assertAlmostEqual(m['boundary_f1'], 1.0)
        self.assertEqual(m['evaluated_positions'], 3)

    def test_unknowns_keep_original_character_and_special_tokens_masked(self):
        vocab = CharVocab(ROOT / 'models/bert-base-chinese')
        text = '甲𰻞A１２3乙'
        ids = vocab.encode(text)
        self.assertEqual(len(ids), len(text))
        self.assertEqual(ids[1], vocab.unk)
        pred = [0]*len(text)
        pred[-1] = 2
        self.assertEqual(render(text, pred), text+'。')
        batch = collate([(ids, [0]*len(ids)), ([1], [2])], vocab)
        self.assertEqual(batch['labels'][0, 0].item(), IGNORE)
        self.assertEqual(batch['labels'][0, len(ids)+1].item(), IGNORE)
        self.assertTrue((batch['labels'][1, 2:] == IGNORE).all())
        self.assertEqual(batch['input_ids'].shape[1] % 8, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
