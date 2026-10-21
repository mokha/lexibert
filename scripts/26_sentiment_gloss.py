"""Sentiment transfer through dictionary glosses, in the spirit of Alnajjar et al. (2023): every word of a Uralic test
sentence is lemmatized with the FST and replaced by an English translation of its lemma from Ve'rdd (directly, or
through Finnish when there is no English link), and the English-trained classifiers of 14_downstream (saved in
work/checkpoints/sentiment) classify the gloss. Words without a translation are kept as they are.
Among several translations, the one with the most dictionary paths is taken, then the most frequent one in the
English training data (SST-2), so the choice does not depend on the test sentences' labels.
Reported: the original sentence, the gloss, and the average of the two predicted distributions.
  python scripts/26_sentiment_gloss.py [encoder names]   -> work/runs_gloss/sentiment[_names].json"""
import collections, json, os, re, sys
import numpy as np, pandas as pd, torch
from datasets import load_from_disk
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
ds = import_module("14_downstream")
W = ds.W
LANGS = ["fin", "kpv", "mdf", "myv", "udm"]
CORE = re.compile(r"[^\W\d_]+(?:[’'ʼʹ-][^\W\d_]+)*|\S")


def dictionary():
    lx = pd.read_parquet(f"{W}/data/verdd/lexemes.parquet", columns=["id", "lang", "lexeme"]).set_index("id")
    rel = pd.read_parquet(f"{W}/data/verdd/relations.parquet", columns=["type", "f", "t"])
    rel = rel[(rel.type == 0) & rel.f.isin(lx.index) & rel.t.isin(lx.index)]
    adj = collections.defaultdict(set)
    for f, t in zip(rel.f, rel.t):
        adj[f].add(t); adj[t].add(f)
    by_text = collections.defaultdict(list)
    for i, l, w in zip(lx.index, lx.lang, lx.lexeme):
        by_text[(l, w.lower())].append(i)
    return lx.lang.to_dict(), lx.lexeme.to_dict(), adj, by_text


def make_gloss(lang_of, text_of, adj, by_text, freq):
    caches = {}

    def lemmas(l, w):
        if l not in caches:
            path = f"{W}/data/fst/{l}.parquet"
            if os.path.exists(path):
                df = pd.read_parquet(path, columns=["word", "analyses"])
                caches[l] = dict(zip(df.word, df.analyses))
            else:  # no cached analyses (Finnish): analyse with UralicNLP
                from uralicNLP import uralicApi
                caches[l] = lambda x: [a for a, _ in uralicApi.analyze(x, l)]
        c = caches[l]
        an = (c(w) or c(w.lower())) if callable(c) else (list(c.get(w, [])) or list(c.get(w.lower(), [])))
        out = [a.split("+")[0] for a in an if "#" not in a]
        return list(dict.fromkeys(out + [w]))  # the form itself last, for words in Ve'rdd as they are

    def translate(l, w):
        votes = collections.Counter()
        for lem in lemmas(l, w):
            for i in by_text.get((l, lem.lower()), []):
                direct = [z for z in adj[i] if lang_of[z] == "eng"]
                for z in direct:
                    votes[text_of[z]] += 2
                if not direct:
                    for f in (z for z in adj[i] if lang_of[z] == "fin"):
                        for z in adj[f]:
                            if lang_of[z] == "eng":
                                votes[text_of[z]] += 1
            if votes:
                break  # the first lemma (in FST order) that has a translation
        if not votes:
            return None
        return max(votes, key=lambda e: (votes[e], freq.get(e.lower(), 0), -len(e), e))

    def gloss(l, sent):
        out = []
        for tok in CORE.findall(sent):
            t = translate(l, tok) if tok[0].isalpha() else None
            out.append(t if t else tok)
        return " ".join(out)
    return gloss


@torch.no_grad()
def probs(model, tok, texts):
    model.eval().cuda()
    e = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.softmax(model(e.input_ids, e.attention_mask).float(), -1).cpu().numpy()


def main():
    sst = load_from_disk(f"{W}/data/sst2")
    freq = collections.Counter(w for s in sst["train"]["sentence"] for w in s.lower().split())
    gloss = make_gloss(*dictionary(), freq)
    test = ds.sentiment_test()
    glossed = {l: [gloss(l, x) for x, _ in test[l]] for l in LANGS}
    os.makedirs(f"{W}/runs_gloss", exist_ok=True)
    json.dump({l: [dict(orig=x, gloss=g, label=y) for (x, y), g in zip(test[l], glossed[l])] for l in LANGS},
              open(f"{W}/runs_gloss/glosses.json", "w"), ensure_ascii=False, indent=1)
    res = collections.defaultdict(dict)
    names = sys.argv[1:] or ["mmbert_base", "pt_mlm", "pt_fst", "pt_full"]
    for name in names:
        for seed in (1, 2, 3):
            ck = torch.load(f"{W}/checkpoints/sentiment/{name}_s{seed}.pt", map_location="cpu", weights_only=False)
            model = ds.Clf(ck["encoder"], ck["steps"]); model.load_state_dict(ck["state"])
            tok = AutoTokenizer.from_pretrained(ck["encoder"])
            r = {}
            for l in LANGS:
                y = np.array([lab for _, lab in test[l]])
                po, pg = probs(model, tok, [x for x, _ in test[l]]), probs(model, tok, glossed[l])
                r[l] = {"orig": round(100 * np.mean(po.argmax(1) == y), 1), "gloss": round(100 * np.mean(pg.argmax(1) == y), 1),
                        "orig+gloss": round(100 * np.mean((po + pg).argmax(1) == y), 1)}
            res[name][seed] = r
            print(name, seed, r, flush=True)
    summary = {name: {l: {k: round(float(np.mean([res[name][s][l][k] for s in (1, 2, 3)])), 1)
                          for k in ("orig", "gloss", "orig+gloss")} for l in LANGS} for name in res}
    print(json.dumps(summary, indent=1))
    json.dump(dict(per_seed=res, mean=summary), open(f"{W}/runs_gloss/sentiment{'_' + '_'.join(names) if sys.argv[1:] else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
