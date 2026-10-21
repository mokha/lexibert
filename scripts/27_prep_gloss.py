"""Word-by-word English glosses for the pretraining corpora (for 06_pretrain --gloss / --align).

Every word type with an FST analysis gets one English translation of its lemma from Ve'rdd: a direct English link,
or a link through Finnish when there is none. Among several translations, the one with the most dictionary paths is
taken, then the most frequent one in the English replay corpus. Words without a translation are glossed as
themselves. 26_sentiment_gloss uses nearly the same rule for the test sentences (tie-break by SST-2 frequency, no
filter on bracketed dictionary entries).
  python scripts/27_prep_gloss.py   -> work/data/pretrain/gloss_{lang}.pkl"""
import collections, os, pickle, re, sys
import pandas as pd

W = os.environ.get("LEXIBERT_WORK", "work")
P = f"{W}/data/pretrain"
LANGS = ["myv", "mdf", "kpv", "koi", "udm", "sme", "sms", "smn", "olo", "krl", "vep", "liv", "mhr", "mrj", "yrk", "mns"]


def main():
    lx = pd.read_parquet(f"{W}/data/verdd/lexemes.parquet", columns=["id", "lang", "lexeme"]).set_index("id")
    rel = pd.read_parquet(f"{W}/data/verdd/relations.parquet", columns=["type", "f", "t"])
    rel = rel[(rel.type == 0) & rel.f.isin(lx.index) & rel.t.isin(lx.index)]
    lang_of, text_of = lx.lang.to_dict(), lx.lexeme.to_dict()
    adj = collections.defaultdict(set)
    for f, t in zip(rel.f, rel.t):
        adj[f].add(t); adj[t].add(f)
    by_text = collections.defaultdict(list)
    for i, l, w in zip(lx.index, lx.lang, lx.lexeme):
        by_text[(l, w.lower())].append(i)
    freq = collections.Counter()
    with open(f"{W}/data/corpus_pivot/eng.txt", encoding="utf-8") as f:
        for line in f:
            freq.update(line.lower().split())

    def translate(l, lemmas):
        for lem in lemmas:
            votes = collections.Counter()
            for i in by_text.get((l, lem.lower()), []):
                direct = [z for z in adj[i] if lang_of[z] == "eng"]
                for z in direct:
                    votes[text_of[z]] += 2
                if not direct:
                    for fi in (z for z in adj[i] if lang_of[z] == "fin"):
                        for z in adj[fi]:
                            if lang_of[z] == "eng":
                                votes[text_of[z]] += 1
            votes = {e: v for e, v in votes.items() if 0 < len(e) < 40 and not re.search(r"[(){}\[\]?;]", e)}
            if votes:
                return max(votes, key=lambda e: (votes[e], freq.get(e.lower(), 0), -len(e), e))
        return None

    for l in LANGS:
        path = f"{W}/data/fst/{l}.parquet"
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path, columns=["word", "analyses"])
        table, cache = {}, {}
        for w, an in zip(df.word, df.analyses):
            lemmas = tuple(dict.fromkeys([a.split("+")[0] for a in an if "#" not in a] + [w]))
            if lemmas not in cache:
                cache[lemmas] = translate(l, lemmas)
            if cache[lemmas]:
                table[w] = cache[lemmas]
        pickle.dump(table, open(f"{P}/gloss_{l}.pkl", "wb"))
        print(l, "word types", len(df), "glossed", len(table), flush=True)


if __name__ == "__main__":
    main()
