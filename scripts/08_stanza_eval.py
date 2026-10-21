"""Evaluate Stanza's released models on gold-tokenised UD test sets (same metrics as 07_ud.py)."""
import json, os, sys
from collections import Counter
import stanza
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
ud = import_module("07_ud")

W = os.environ.get("LEXIBERT_WORK", "work")
STANZA_LANG = {"myv": "myv", "sme": "sme", "kpv_lattice": "kpv", "kpv_ikdp": "kpv", "fin": "fi", "est": "et"}


def main():
    res = {}
    for tb, lang in STANZA_LANG.items():
        sents = ud.split(tb, "test")
        nlp = stanza.Pipeline(lang, dir=f"{W}/stanza", processors="tokenize,pos,lemma,depparse",
                              tokenize_pretokenized=True, logging_level="WARN", download_method=None)
        doc = nlp([[t["form"] for t in s] for s in sents])
        c = Counter()
        for s, ps in zip(sents, doc.sentences):
            for t, w in zip(s, ps.words):
                c["n"] += 1
                c["upos"] += w.upos == t["upos"]
                c["feats"] += (w.feats or "_") == t["feats"]
                c["lemma"] += (w.lemma or "").lower() == t["lemma"].lower()
                c["uas"] += w.head == t["head"]
                c["las"] += w.head == t["head"] and w.deprel.split(":")[0] == t["rel"]
        res[tb] = {k: round(100 * c[k] / c["n"], 2) for k in ["upos", "feats", "lemma", "uas", "las"]}
        print(tb, res[tb], flush=True)
    os.makedirs(f"{W}/runs/ud", exist_ok=True)
    json.dump(dict(results=res), open(f"{W}/runs/ud/stanza_released.json", "w"), indent=1)


if __name__ == "__main__":
    main()
