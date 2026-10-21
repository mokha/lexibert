"""Re-evaluate the saved UD parsers (07_ud, 13_ud_static) on the test sets without retraining.

Used after the decoding was changed to the highest scoring tree with one root (Chu-Liu-Edmonds) and the FST lemma
choice was made deterministic. Results are written to work/runs_mst/{ud,ud_static}/ in the same format as the
original runs, so that the table scripts can read them.
  python scripts/23_reeval.py ud            [checkpoint names...]
  python scripts/23_reeval.py ud_static     [checkpoint names...]"""
import glob, json, os, sys
from importlib import import_module
import torch

sys.path.insert(0, os.path.dirname(__file__))
ud = import_module("07_ud")
W = ud.W
OUT = f"{W}/{os.environ.get('RUNS', 'runs_mst')}"


def vocab(itos):
    v = ud.Vocab([])
    v.itos, v.stoi = list(itos), {x: i for i, x in enumerate(itos)}
    return v


def tests_of(args):
    return [args["train"][0]] if args["setting"] == "mono" else [tb for tb in ud.TB if tb not in ud.SOURCE_CAP]


def reeval_ud(path):
    from transformers import AutoTokenizer
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a, v = ck["args"], {k: vocab(x) for k, x in ck["vocabs"].items()}
    model = ud.Parser(a["encoder"], v, steps=a["steps"])
    model.load_state_dict(ck["state"])
    tok = AutoTokenizer.from_pretrained(a["encoder"])
    res = {}
    for tb in tests_of(a):
        ts = ud.split(tb, "test")
        res[tb] = ud.evaluate(model, ts, ud.encode(ts, tok, v), v, tok.pad_token_id, "cuda", tb)
    return dict(args=a, results=res)


def reeval_static(path):
    st = import_module("13_ud_static")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a, v = ck["args"], {k: vocab(x) for k, x in ck["vocabs"].items()}
    wv, cv = vocab(ck["words"]), vocab(ck["chars"])
    tests = tests_of(a)
    train_tbs = [a["train"][0]] if a["setting"] == "mono" else list(ud.TB)
    src = st.Source(a["vectors"], [st.TB_LANG[tb] for tb in train_tbs + tests])
    model = st.Tagger(len(wv.itos), len(cv.itos), src.dim, {k: len(x.itos) for k, x in v.items()}, a["steps"])
    model.load_state_dict(ck["state"])
    res = {}
    for tb in tests:
        ts = ud.split(tb, "test")
        res[tb] = st.evaluate(model, ts, st.build({tb: ts}, src, wv, cv, v), v, tb) | {"vec_coverage": src.coverage(ts, st.TB_LANG[tb])}
    return dict(args=a, results=res)


def main():
    kind, names = sys.argv[1], sys.argv[2:]
    fn = {"ud": reeval_ud, "ud_static": reeval_static}[kind]
    os.makedirs(f"{OUT}/{kind}", exist_ok=True)
    for path in sorted(glob.glob(f"{W}/checkpoints/{kind}/*.pt")):
        name = os.path.basename(path)[:-3]
        out = f"{OUT}/{kind}/{name}.json"
        if (names and name not in names) or os.path.exists(out):
            continue
        r = fn(path)
        json.dump(r, open(out, "w"), indent=1)
        print(name, {tb: (x["las"], x["lemma_fst"]) for tb, x in r["results"].items()}, flush=True)


if __name__ == "__main__":
    main()
