"""Prepare lookups for lexicon-grounded continued pretraining:
  - tag vocabulary + per-word soft FST tag targets      (FST objective)
  - word form -> translations (eng/fin/rus/est)          (code-switching objective)
  - Ve'rdd example-sentence pairs (mhr-rus, kpv-rus)     (translation-LM objective)
Only the current Ve'rdd relations are used (relations.parquet)."""
import collections, json, os, pickle, random, re
import pandas as pd

W = os.environ.get("LEXIBERT_WORK", "work")
OUT = f"{W}/data/pretrain"
LANGS = ["myv", "mdf", "kpv", "koi", "udm", "sme", "sms", "smn", "olo", "krl", "vep", "liv", "mhr", "mrj", "yrk", "mns"]
PIVOTS = {"eng", "fin", "rus", "est"}
MIN_TAG_FREQ = 50
RETRIEVAL_TEST = 500


def tags_of(analysis):
    if "+Err" in analysis:
        return None
    analysis = analysis.split("#")[-1]  # compounds: take the head
    parts = [p for p in analysis.split("+")[1:] if p]
    return parts or None


def main():
    os.makedirs(OUT, exist_ok=True)
    random.seed(0)
    # ---- FST tag targets
    counts, per_lang = collections.Counter(), {}
    for lang in LANGS:
        df = pd.read_parquet(f"{W}/data/fst/{lang}.parquet")
        df = df[df.analyses.map(len) > 0]
        per_lang[lang] = df
        for analyses, freq in zip(df.analyses, df.freq):
            for a in analyses:
                for t in tags_of(a) or []:
                    counts[t] += 1
    vocab = [t for t, c in counts.most_common() if c >= MIN_TAG_FREQ]
    idx = {t: i for i, t in enumerate(vocab)}
    json.dump(vocab, open(f"{OUT}/tags.json", "w"), ensure_ascii=False)
    print("tag vocab", len(vocab))
    lemma_of = {}
    for lang, df in per_lang.items():
        table = {}
        for word, analyses in zip(df.word, df.analyses):
            parsed = [p for p in map(tags_of, analyses) if p]
            if not parsed:
                continue
            soft = collections.Counter()
            for tags in parsed:
                for t in set(tags):
                    if t in idx:
                        soft[idx[t]] += 1 / len(parsed)
            table[word] = sorted(soft.items())
            lemma_of[(lang, word)] = {a.split("+")[0] for a in analyses if "#" not in a}
        pickle.dump(table, open(f"{OUT}/fst_{lang}.pkl", "wb"))
        print(lang, "fst words", len(table), flush=True)
    # ---- code-switching dictionary through Ve'rdd translations
    lx = pd.read_parquet(f"{W}/data/verdd/lexemes.parquet", columns=["id", "lang", "lexeme"])
    rel = pd.read_parquet(f"{W}/data/verdd/relations.parquet", columns=["type", "f", "t"])
    rel = rel[rel.type == 0]
    lang_of, text_of = lx.set_index("id").lang, lx.set_index("id").lexeme
    trans = collections.defaultdict(set)
    for a, b in zip(rel.f, rel.t):
        la, lb = lang_of.get(a), lang_of.get(b)
        if la in LANGS and lb in PIVOTS:
            trans[(la, text_of[a])].add(text_of[b])
        elif lb in LANGS and la in PIVOTS:
            trans[(lb, text_of[b])].add(text_of[a])
    cs = collections.defaultdict(dict)
    for (lang, word), lemmas in lemma_of.items():
        options = set().union(*(trans.get((lang, l), set()) for l in lemmas))
        options = [o for o in options if 0 < len(o) < 40]
        if options:
            cs[lang][word] = sorted(options)[:8]
    pickle.dump(dict(cs), open(f"{OUT}/codeswitch.pkl", "wb"))
    print("code-switch coverage (word types):", {l: len(v) for l, v in cs.items()})
    # ---- parallel example pairs from Ve'rdd relation examples
    ex = pd.read_parquet(f"{W}/data/verdd/relation_examples.parquet")
    pairs = []
    for rid, g in ex.groupby("relation_id"):
        langs = dict(zip(g.lang, g.text))
        for src in ("mhr", "kpv"):
            if src in langs and "rus" in langs and len(g) == 2:
                pairs.append((src, langs[src].strip(), langs["rus"].strip()))
    random.shuffle(pairs)
    kpv = [p for p in pairs if p[0] == "kpv"]
    test = kpv[:RETRIEVAL_TEST]
    train = [p for p in pairs if p not in set(test)]
    json.dump(test, open(f"{OUT}/kpv_rus_retrieval_test.json", "w"), ensure_ascii=False)
    json.dump(train, open(f"{OUT}/parallel_train.json", "w"), ensure_ascii=False)
    print("parallel train", collections.Counter(p[0] for p in train), "kpv-rus retrieval test", len(test))


if __name__ == "__main__":
    main()
