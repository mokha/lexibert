"""Sub-word tokens per word of the mmBERT-small tokenizer in our corpora (Section 3 of the paper): words made of
letters only, tokenized in their sentences (as the models see them), in 20,000 random sentences per language."""
import collections, os, random, re
from transformers import AutoTokenizer

W = os.environ.get("LEXIBERT_WORK", "work")
LANGS = ["myv", "mdf", "kpv", "koi", "udm", "sme", "sms", "smn", "olo", "krl", "vep", "liv", "mhr", "mrj", "yrk"]
CORE = re.compile(r"[^\W\d_]+(?:[’'ʼʹ-][^\W\d_]+)*")

tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-small")
res = {}
for lang in LANGS:
    lines = open(f"{W}/data/corpus/{lang}.txt", encoding="utf-8").read().splitlines()
    sample = random.Random(0).sample(lines, min(20000, len(lines)))
    tokens = words = 0
    for i in range(0, len(sample), 500):
        batch = [s.split() for s in sample[i:i + 500] if s.split()]
        enc = tok(batch, is_split_into_words=True, add_special_tokens=False)
        for k, ws in enumerate(batch):
            count = collections.Counter(w for w in enc.word_ids(k) if w is not None)
            for j, w in enumerate(ws):
                if CORE.fullmatch(w):
                    tokens += count[j]; words += 1
    res[lang] = tokens / words
    print(lang, round(res[lang], 2), flush=True)
print("mean", round(sum(res.values()) / len(res), 2), "min", round(min(res.values()), 1), "max", round(max(res.values()), 1))
