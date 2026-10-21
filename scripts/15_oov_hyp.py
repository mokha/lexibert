"""Write unknown-word-model hypotheses ("lemma+POS+tags") for FST-unknown corpus types,
used by morph_embeddings as analyses for words the FST cannot analyse."""
import os, sys
import pandas as pd, torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
m10 = import_module("10_oov_model")
oov = import_module("04_build_oov_data")

W = os.environ.get("LEXIBERT_WORK", "work")
MIN_P, CAP = 0.5, 100_000


def main(name):
    ck = torch.load(f"{W}/models/oov_{name}/model.pt", weights_only=False)
    voc = {k: m10.Vocab([]) for k in ck["vocabs"]}
    for k, itos in ck["vocabs"].items():
        voc[k].itos, voc[k].stoi = itos, {x: i for i, x in enumerate(itos)}
    model = m10.OOVModel(ck["encoder"], {k: len(x) for k, x in ck["vocabs"].items()})
    model.load_state_dict(ck["state"])
    tok = AutoTokenizer.from_pretrained(ck["encoder"])
    out = f"{W}/data/oov_hyp/{name}"
    os.makedirs(out, exist_ok=True)
    for lang in m10.LANGS:
        p = f"{W}/data/fst/{lang}.parquet"
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        unk = df[(df.analyses.map(len) == 0) & (df.freq > 0)].sort_values("freq", ascending=False).head(CAP)
        if not len(unk):
            continue
        pred = m10.predict(model, tok, [(lang, w) for w in unk.word], k=1)
        rows, j = [], 0
        for chunk in pred:
            for b in range(len(chunk["rule"][1])):
                w = unk.word.iloc[j]; j += 1
                r, pr = chunk["rule"][1][b, 0].item(), chunk["rule"][0][b, 0].item()
                lemma = oov.apply_rule(w, voc["rule"].itos[r]) if r else None
                tags = voc["tags"].itos[chunk["tags"][1][b, 0]]
                if lemma and pr >= MIN_P and "|" in tags:
                    pos, rest = tags.split("|", 1)
                    rows.append(dict(word=w, analysis="+".join([lemma, pos] + [t for t in rest.split("+") if t]), p=pr))
        pd.DataFrame(rows).to_parquet(f"{out}/{lang}.parquet", index=False)
        print(lang, "unknown types", len(unk), "hypotheses kept", len(rows), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
