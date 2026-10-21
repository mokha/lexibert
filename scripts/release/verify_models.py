"""Check that the release folders reproduce the results of the paper (UD parsers, unknown word model, sentiment)."""
import json, os, sys
from collections import Counter
from importlib import import_module
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
ud = import_module("07_ud")
from lexibert import Parser, UnknownWordModel, SentimentClassifier

W = os.environ.get("LEXIBERT_WORK", "work")
M = f"{W}/release/models"


def check_ud(name, run, tbs):
    p = Parser.from_pretrained(f"{M}/{name}")
    ref = json.load(open(f"{W}/runs/ud/{run}.json"))["results"]
    for tb in tbs:
        sents = ud.split(tb, "test")
        out = p.parse([[t["form"] for t in s] for s in sents])
        out_fst = p.parse([[t["form"] for t in s] for s in sents], fst_lang=ud.FST_LANG[tb])
        c = Counter()
        for s, o, of in zip(sents, out, out_fst):
            for t, x, xf in zip(s, o, of):
                c["n"] += 1
                c["upos"] += x["upos"] == t["upos"]; c["feats"] += x["feats"] == t["feats"]
                c["lemma"] += x["lemma"].lower() == t["lemma"].lower()
                c["lemma_fst"] += xf["lemma"].lower() == t["lemma"].lower()
                c["uas"] += x["head"] == t["head"]; c["las"] += x["head"] == t["head"] and x["deprel"] == t["rel"]
        got = {k: round(100 * c[k] / c["n"], 2) for k in ["upos", "feats", "lemma", "lemma_fst", "uas", "las"]}
        diff = {k: round(got[k] - ref[tb][k], 2) for k in got}
        print(name, tb, got, "diff", diff, flush=True)


def check_oov():
    m = UnknownWordModel.from_pretrained(f"{M}/lexibert-unknown-words")
    pred = pd.read_parquet(f"{W}/runs/oov/pt_gloss_predictions.parquet")
    same_l = same_c = n = 0
    for lang, df in pred.groupby("lang"):
        df = df.head(1000)
        res = m.analyse(lang, list(df.form))
        for r, (_, row) in zip(res, df.iterrows()):
            n += 1
            same_l += (r["lemma"][0][0] if r["lemma"] else None) == row.pred_lemma
            same_c += row.contlex is None or not row.contlex or (bool(r["contlex"]) and r["contlex"][0][0] == row.contlex) == bool(row.contlex_hit)
    print(f"unknown words: same first lemma {100 * same_l / n:.2f}%, same contlex hit {100 * same_c / n:.2f}% of {n}", flush=True)


def check_sentiment():
    ds = import_module("14_downstream")
    m = SentimentClassifier.from_pretrained(f"{M}/lexibert-sentiment-sst2")
    for lang, rows in ds.sentiment_test().items():
        pred = m.predict([x for x, _ in rows])
        print("sentiment", lang, round(100 * sum((p == "positive") == bool(y) for p, (_, y) in zip(pred, rows)) / len(rows), 1), flush=True)


if __name__ == "__main__":
    what = sys.argv[1:] or ["ud", "oov", "sentiment"]
    if "ud" in what:
        check_ud("lexibert-ud-myv", "pt_gloss_myv_mono_s1", ["myv"])
        check_ud("lexibert-ud-sme", "pt_gloss_sme_mono_s1", ["sme"])
        check_ud("lexibert-ud-uralic", "pt_gloss_multi_s1", ud.TB.keys() - ud.SOURCE_CAP.keys())
    if "oov" in what:
        check_oov()
    if "sentiment" in what:
        check_sentiment()
