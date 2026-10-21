"""FST-analyse every word type of each corpus (+ UD tokens) with C-pyhfst; cache to parquet.
Incremental: types already in the cache are skipped."""
import collections, glob, os, re, sys
from multiprocessing import Pool
import pandas as pd

W = os.environ.get("LEXIBERT_WORK", "work")
OUT = f"{W}/data/fst"
TOKEN = re.compile(r"[^\W\d_]+(?:[’'ʼʹ-][^\W\d_]+)*")
UD = {"myv": "Erzya-JR", "mdf": "Moksha-JR", "kpv": "Komi_Zyrian-*", "koi": "Komi_Permyak-UH",
      "sms": "Skolt_Sami-Giellagas", "olo": "Livvi-KKPP", "krl": "Karelian-KKPP", "sme": "North_Sami-Giella",
      "vep": "Veps-VWT", "yrk": "Nenets-Tundra"}
_lang = None


def _init(lang):
    global _lang
    _lang = lang
    from uralicNLP import uralicApi
    globals()["uapi"] = uralicApi


def _analyze(word):
    res = uapi.analyze(word, _lang)
    lowered = False
    if not res and word[:1].isupper():
        res = uapi.analyze(word.lower(), _lang)
        lowered = bool(res)
    return word, [a for a, _ in res], lowered


def types_for(lang):
    c = collections.Counter()
    path = f"{W}/data/corpus/{lang}.txt"
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            c.update(TOKEN.findall(line))
    for f in glob.glob(f"{W}/data/ud/UD_{UD.get(lang, '__none__')}/*.conllu"):
        for line in open(f, encoding="utf-8"):
            parts = line.split("\t")
            if len(parts) == 10 and parts[0].isdigit():
                c[parts[1]] += 0  # make sure UD forms are analysed, without inflating corpus counts
    return c


def main(langs, procs=30):
    os.makedirs(OUT, exist_ok=True)
    for lang in langs:
        cache_path = f"{OUT}/{lang}.parquet"
        cache = pd.read_parquet(cache_path) if os.path.exists(cache_path) else None
        counts = types_for(lang)
        cached = set() if cache is None else set(cache.word)
        todo = [w for w in counts if w not in cached]
        with Pool(procs, initializer=_init, initargs=(lang,)) as pool:
            rows = pool.map(_analyze, todo, chunksize=500)
        df = pd.DataFrame(rows, columns=["word", "analyses", "lowered"])
        if cache is not None:
            df = pd.concat([cache.drop(columns=["freq"]), df], ignore_index=True)
        df["freq"] = df.word.map(counts).fillna(0).astype(int)
        df.to_parquet(cache_path, index=False)
        known = df.analyses.map(len) > 0
        tok_cov = df.freq[known].sum() / max(df.freq.sum(), 1)
        print(f"{lang}: types={len(df)} type_cov={known.mean():.3f} token_cov={tok_cov:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
