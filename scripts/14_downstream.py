"""Downstream evaluation of an encoder or a static embedding set.

  sentiment  train on English SST-2 only, test zero-shot on the aligned 68-sentence fin/kpv/mdf/myv/udm set
             (Alnajjar et al. 2023). Encoders: fine-tuned, 3 seeds. Static: mean vector + logistic regression.
  retrieval  kpv->rus and rus->kpv P@1 on 500 held-out Ve'rdd example pairs (no training)."""
import argparse, json, os, pickle, re, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import lightning as L
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module

W = os.environ.get("LEXIBERT_WORK", "work")
RUNS = os.environ.get("RUNS", "runs")  # results dir under work/ (reruns write elsewhere so they can be compared)
CKPT = f"{W}/checkpoints"
NAME = None  # set in main(); used for checkpoint file names
SENT_LANGS = ["fin", "kpv", "mdf", "myv", "udm"]
CORE = re.compile(r"[^\W\d_]+(?:[’'ʼʹ-][^\W\d_]+)*")


def sentiment_test():
    d = json.load(open(f"{W}/downloads/zenodo_7866456/sentiment_eval_data.json"))
    return {l: [(x[l], int(lab == "positive")) for lab, xs in d.items() for x in xs if l in x] for l in SENT_LANGS}


class Clf(L.LightningModule):
    def __init__(self, encoder, steps):
        super().__init__()
        self.save_hyperparameters()
        self.enc = AutoModel.from_pretrained(encoder)
        self.out = nn.Linear(self.enc.config.hidden_size, 2)

    def forward(self, ids, att):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        return self.out((h * att[..., None]).sum(1) / att.sum(1, keepdim=True))

    def training_step(self, b, _):
        loss = F.cross_entropy(self(b[0], b[1]).float(), b[2])
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=2e-5, weight_decay=0.01)
        s = self.hparams.steps
        return [opt], [{"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / (0.1 * s)) * max(0, 1 - i / s)), "interval": "step"}]


@torch.no_grad()
def predict(model, tok, texts):
    model.eval().cuda()
    out = []
    for i in range(0, len(texts), 64):
        e = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out += model(e.input_ids, e.attention_mask).argmax(-1).tolist()
    return out


def sentiment_encoder(encoder, seeds, steps=2000):
    sst = load_from_disk(f"{W}/data/sst2")
    tok = AutoTokenizer.from_pretrained(encoder)
    test = sentiment_test()
    res = {}
    for seed in seeds:
        L.seed_everything(seed)
        rows = list(zip(sst["train"]["sentence"], sst["train"]["label"]))

        def coll(b):
            e = tok([x for x, _ in b], padding=True, truncation=True, max_length=96, return_tensors="pt")
            return e.input_ids, e.attention_mask, torch.tensor([y for _, y in b])

        model = Clf(encoder, steps)
        L.Trainer(max_steps=steps, precision="bf16-mixed", enable_checkpointing=False, logger=False,
                  enable_progress_bar=False).fit(model, DataLoader(rows, batch_size=32, shuffle=True, collate_fn=coll, num_workers=4))
        os.makedirs(f"{CKPT}/sentiment", exist_ok=True)
        torch.save({"state": model.state_dict(), "encoder": encoder, "steps": steps}, f"{CKPT}/sentiment/{NAME}_s{seed}.pt")
        r = {"eng(sst2-dev)": np.mean(np.array(predict(model, tok, list(sst["validation"]["sentence"]))) == np.array(sst["validation"]["label"]))}
        for l, xs in test.items():
            r[l] = np.mean(np.array(predict(model, tok, [x for x, _ in xs])) == np.array([y for _, y in xs]))
        res[seed] = {k: round(100 * v, 1) for k, v in r.items()}
        print("sentiment seed", seed, res[seed], flush=True)
    return res


def mean_vec(get, lang, text, dim):
    vs = [v for w in CORE.findall(text) if (v := get(lang, w)) is not None]
    return np.mean(vs, 0) if vs else np.zeros(dim, np.float32)


def sentiment_static(emb):
    from sklearn.linear_model import LogisticRegression
    ud_static = import_module("13_ud_static")
    src = ud_static.Source(f"morph:{emb}", SENT_LANGS + ["eng"])
    sst = load_from_disk(f"{W}/data/sst2")
    X = np.stack([mean_vec(src.get, "eng", s, src.dim) for s in sst["train"]["sentence"]])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, sst["train"]["label"])
    os.makedirs(f"{CKPT}/sentiment", exist_ok=True)
    pickle.dump(clf, open(f"{CKPT}/sentiment/{NAME}_static.pkl", "wb"))
    r = {"eng(sst2-dev)": clf.score(np.stack([mean_vec(src.get, "eng", s, src.dim) for s in sst["validation"]["sentence"]]), sst["validation"]["label"])}
    for l, xs in sentiment_test().items():
        r[l] = clf.score(np.stack([mean_vec(src.get, l, x, src.dim) for x, _ in xs]), [y for _, y in xs])
    r = {k: round(100 * v, 1) for k, v in r.items()}
    print("sentiment static", r, flush=True)
    return r


def procrustes_maps(src, langs, target="eng", max_pairs=20000):
    """Orthogonal map per language from its vector space to the target's, learned from live Ve'rdd translations
    (single words only), as in the supervised alignment of Alnajjar et al. (2023)."""
    import pandas as pd
    lx = pd.read_parquet(f"{W}/data/verdd/lexemes.parquet", columns=["id", "lang", "lexeme"]).set_index("id")
    rel = pd.read_parquet(f"{W}/data/verdd/relations.parquet", columns=["type", "f", "t"])
    rel = rel[rel.type == 0]
    lang, text = rel.f.map(lx.lang), rel.t.map(lx.lang)
    maps = {}
    for l in langs:
        if l == target:
            continue
        a = rel[(lang == l) & (text == target)]; b = rel[(lang == target) & (text == l)]
        pairs = list(zip(a.f.map(lx.lexeme), a.t.map(lx.lexeme))) + list(zip(b.t.map(lx.lexeme), b.f.map(lx.lexeme)))
        pairs = [(x, y) for x, y in dict.fromkeys(pairs) if x.isalpha() and y.isalpha()][:max_pairs]
        X, Y = [], []
        for x, y in pairs:
            vx, vy = src.get(l, x), src.get(target, y)
            if vx is not None and vy is not None:
                X.append(vx); Y.append(vy)
        if len(X) < 50:
            continue
        u, _, vt = np.linalg.svd(np.asarray(X).T @ np.asarray(Y))
        maps[l] = u @ vt
        print("procrustes", l, "pairs", len(X), flush=True)
    return maps


def sentiment_static_aligned(emb):
    from sklearn.linear_model import LogisticRegression
    ud_static = import_module("13_ud_static")
    src = ud_static.Source(f"morph:{emb}", SENT_LANGS + ["eng"])
    maps = procrustes_maps(src, SENT_LANGS)
    get = lambda l, w: (v @ maps[l] if l in maps else v) if (v := src.get(l, w)) is not None else None
    sst = load_from_disk(f"{W}/data/sst2")
    X = np.stack([mean_vec(get, "eng", s, src.dim) for s in sst["train"]["sentence"]])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, sst["train"]["label"])
    os.makedirs(f"{CKPT}/sentiment", exist_ok=True)
    pickle.dump({"clf": clf, "maps": maps}, open(f"{CKPT}/sentiment/{NAME}_static_aligned.pkl", "wb"))
    r = {"eng(sst2-dev)": clf.score(np.stack([mean_vec(get, "eng", s, src.dim) for s in sst["validation"]["sentence"]]), sst["validation"]["label"])}
    for l, xs in sentiment_test().items():
        r[l] = clf.score(np.stack([mean_vec(get, l, x, src.dim) for x, _ in xs]), [y for _, y in xs])
    r = {k: round(100 * v, 1) for k, v in r.items()}
    print("sentiment static aligned", r, flush=True)
    return r


def p_at_1(a, b):
    a, b = F.normalize(torch.tensor(np.asarray(a)), dim=-1), F.normalize(torch.tensor(np.asarray(b)), dim=-1)
    s = a @ b.T
    gold = torch.arange(len(a))
    return round(100 * (s.argmax(1) == gold).float().mean().item(), 1), round(100 * (s.argmax(0) == gold).float().mean().item(), 1)


@torch.no_grad()
def retrieval_encoder(encoder, layers=(8, 16)):
    pairs = json.load(open(f"{W}/data/pretrain/kpv_rus_retrieval_test.json"))
    tok = AutoTokenizer.from_pretrained(encoder)
    enc = AutoModel.from_pretrained(encoder, dtype=torch.bfloat16).cuda().eval()

    def emb(texts):
        out = []
        for i in range(0, len(texts), 64):
            e = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            hs = enc(**e, output_hidden_states=True).hidden_states
            h = torch.stack(hs[layers[0]:layers[1] + 1]).float().mean(0)
            m = e.attention_mask[..., None].float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu())
        return torch.cat(out)

    a, b = emb([p[1] for p in pairs]), emb([p[2] for p in pairs])
    r = dict(zip(["kpv->rus", "rus->kpv"], p_at_1(a, b)))
    print("retrieval encoder", r, flush=True)
    return r


def retrieval_static(emb):
    ud_static = import_module("13_ud_static")
    src = ud_static.Source(f"morph:{emb}", ["kpv", "rus"])
    pairs = json.load(open(f"{W}/data/pretrain/kpv_rus_retrieval_test.json"))
    a = [mean_vec(src.get, "kpv", p[1], src.dim) for p in pairs]
    b = [mean_vec(src.get, "rus", p[2], src.dim) for p in pairs]
    r = dict(zip(["kpv->rus", "rus->kpv"], p_at_1(a, b)))
    print("retrieval static", r, flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder")
    ap.add_argument("--emb")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tasks", nargs="+", default=["sentiment", "retrieval"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    global NAME
    NAME = a.name
    torch.set_float32_matmul_precision("high")
    res = {}
    if a.encoder:
        if "sentiment" in a.tasks:
            res["sentiment"] = sentiment_encoder(a.encoder, a.seeds, a.steps)
        if "retrieval" in a.tasks:
            res["retrieval"] = retrieval_encoder(a.encoder)
    if a.emb:
        out = f"{W}/{RUNS}/downstream/{a.name}.json"
        if os.path.exists(out):  # extend an existing result file instead of overwriting it
            res = json.load(open(out))
        if "sentiment_aligned" in a.tasks:
            res["sentiment_static_aligned"] = sentiment_static_aligned(a.emb)
        if "sentiment" in a.tasks:
            res["sentiment_static"] = sentiment_static(a.emb)
        if "retrieval" in a.tasks:
            res["retrieval_static"] = retrieval_static(a.emb)
    os.makedirs(f"{W}/{RUNS}/downstream", exist_ok=True)
    json.dump(res, open(f"{W}/{RUNS}/downstream/{a.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
