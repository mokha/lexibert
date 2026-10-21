"""Unknown-word analysis: word form -> ranked (lemma, POS, tags, contlex) hypotheses.

One multilingual encoder with four classification heads: lemma edit rule, POS, tag string, contlex.
Evaluated on held-out lemmas (grouped split) and on real FST-unknown UD test tokens."""
import argparse, json, os, sys, random
from collections import Counter
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
oov = import_module("04_build_oov_data")

W = os.environ.get("LEXIBERT_WORK", "work")
LANGS = ["myv", "mdf", "kpv", "koi", "udm", "sme", "sms", "smn", "olo", "krl", "vep", "liv", "mhr", "mrj", "yrk"]
PER_LANG = 60000
UPOS = {"N": "NOUN", "V": "VERB", "A": "ADJ", "Adv": "ADV", "Pron": "PRON", "Num": "NUM", "Adp": "ADP", "Po": "ADP",
        "Pr": "ADP", "CC": "CCONJ", "CS": "SCONJ", "Pcle": "PART", "Interj": "INTJ", "Det": "DET", "Prop": "PROPN"}


class Vocab:
    def __init__(self, items, min_count=2):
        c = Counter(items)
        self.itos = ["<unk>"] + [x for x, n in c.most_common() if n >= min_count]
        self.stoi = {x: i for i, x in enumerate(self.itos)}

    def __call__(self, x):
        return self.stoi.get(x, 0)


class OOVModel(L.LightningModule):
    def __init__(self, encoder, sizes, lr=5e-5, steps=6000):
        super().__init__()
        self.save_hyperparameters()
        self.enc = AutoModel.from_pretrained(encoder)
        H = self.enc.config.hidden_size
        self.heads = nn.ModuleDict({k: nn.Sequential(nn.Dropout(0.1), nn.Linear(H, n)) for k, n in sizes.items()})

    def forward(self, ids, att):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        pooled = (h * att[..., None]).sum(1) / att.sum(1, keepdim=True)
        return {k: head(pooled) for k, head in self.heads.items()}

    def training_step(self, batch, _):
        ids, att, y = batch
        out = self(ids, att)
        loss = sum(F.cross_entropy(out[k].float(), y[k], ignore_index=0) for k in out)
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        enc = [p for n, p in self.named_parameters() if n.startswith("enc.")]
        rest = [p for n, p in self.named_parameters() if not n.startswith("enc.")]
        opt = torch.optim.AdamW([{"params": enc, "lr": self.hparams.lr}, {"params": rest, "lr": 1e-3}], weight_decay=0.01)
        s = self.hparams.steps
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / (0.06 * s)) * max(0.0, (s - i) / s))
        return [opt], [{"scheduler": sched, "interval": "step"}]


def text(lang, form):
    return f"{lang}: {form}"


def collate(rows, tok, v):
    enc = tok([text(r["lang"], r["form"]) for r in rows], padding=True, truncation=True, max_length=32, return_tensors="pt")
    y = {"rule": torch.tensor([v["rule"](r["rule"]) for r in rows]), "pos": torch.tensor([v["pos"](r["pos"]) for r in rows]),
         "tags": torch.tensor([v["tags"](r["pos"] + "|" + r["tags"]) for r in rows]),
         "contlex": torch.tensor([v["contlex"](r["lang"] + "|" + r["contlex"]) if r["contlex"] else 0 for r in rows])}
    return enc.input_ids, enc.attention_mask, y


@torch.no_grad()
def predict(model, tok, langs_forms, k=5, bs=256):
    model.eval().cuda()
    outs = []
    for i in range(0, len(langs_forms), bs):
        chunk = langs_forms[i:i + bs]
        enc = tok([text(l, f) for l, f in chunk], padding=True, truncation=True, max_length=32, return_tensors="pt").to("cuda")
        o = model(enc.input_ids, enc.attention_mask)
        outs.append({h: o[h].float().softmax(-1).topk(k) for h in o})
        outs[-1] = {h: (t.values.cpu(), t.indices.cpu()) for h, t in outs[-1].items()}
    return outs


def calibrate(model, tok, v, ud_train):
    """Pick tau: copy the form as lemma when the top rule probability is below tau (tuned on UD *train* OOVs)."""
    if not len(ud_train):
        return 0.0
    pred = predict(model, tok, list(zip(ud_train.lang, ud_train.form)))
    probs, hits_model, hits_copy = [], [], []
    j = 0
    for chunk in pred:
        for b in range(len(chunk["rule"][1])):
            row = ud_train.iloc[j]; j += 1
            r = chunk["rule"][1][b, 0].item()
            lemma = oov.apply_rule(row.form, v["rule"].itos[r]) if r else None
            probs.append(chunk["rule"][0][b, 0].item())
            hits_model.append((lemma or "").lower() == row.lemma.lower())
            hits_copy.append(row.form.lower() == row.lemma.lower())
    best = max((sum(hm if p >= t else hc for p, hm, hc in zip(probs, hits_model, hits_copy)), t)
               for t in [i / 20 for i in range(21)])
    print("calibration on", len(probs), "UD-train OOV tokens: tau", best[1], "acc", round(100 * best[0] / len(probs), 1), flush=True)
    return best[1]


def bucket(n):
    return "unseen" if n == 0 else "rare(1-10)" if n <= 10 else "mid(11-100)" if n <= 100 else "frequent(>100)"


def evaluate(model, tok, v, test, ud, tau=0.0, cl_freq=None):
    res, strata, rows_out = {}, {}, []
    cl_freq = cl_freq or {}
    for lang, df in test.groupby("lang"):
        forms = df.drop_duplicates("form")
        pred = predict(model, tok, list(zip(forms.lang, forms.form)))
        c = Counter()
        j = 0
        for chunk in pred:
            for b in range(len(chunk["rule"][1])):
                row = forms.iloc[j]; j += 1
                gold = [tuple(x) for x in json.loads(row.gold)]
                gl = {g[0] for g in gold}
                lemmas = [oov.apply_rule(row.form, v["rule"].itos[r]) if r else None for r in chunk["rule"][1][b].tolist()]
                pos = v["pos"].itos[chunk["pos"][1][b, 0]]
                tags = v["tags"].itos[chunk["tags"][1][b, 0]]
                c["n"] += 1
                hit = {"lemma@1": lemmas[0] in gl, "lemma@5": any(l in gl for l in lemmas),
                       "pos": pos in {g[1] for g in gold},
                       "full@1": (lemmas[0], *tags.split("|", 1)) in set(gold) if "|" in tags else False}
                for k, x in hit.items():
                    c[k] += x
                cl_hit = None
                if row.contlex:
                    cl_hit = v["contlex"].itos[chunk["contlex"][1][b, 0]] == lang + "|" + row.contlex
                keys = [("pos", row.pos)]
                if row.contlex:
                    keys.append(("contlex_class", bucket(cl_freq.get(lang + "|" + row.contlex, 0))))
                for key in keys:
                    sc = strata.setdefault(key, Counter())
                    sc["n"] += 1
                    for k, x in hit.items():
                        sc[k] += x
                    if cl_hit is not None:
                        sc["n_cl"] += 1; sc["contlex@1"] += cl_hit
                rows_out.append(dict(lang=lang, form=row.form, gold=row.gold, pred_lemma=lemmas[0], pred_pos=pos,
                                     pred_tags=tags, contlex=row.contlex, lemma_hit=hit["lemma@1"], contlex_hit=cl_hit))
                if row.contlex:
                    c["n_cl"] += 1
                    c["contlex@1"] += v["contlex"].itos[chunk["contlex"][1][b, 0]] == lang + "|" + row.contlex
                    c["contlex@3"] += lang + "|" + row.contlex in {v["contlex"].itos[x] for x in chunk["contlex"][1][b, :3].tolist()}
        r = {k: round(100 * c[k] / c["n"], 1) for k in ["lemma@1", "lemma@5", "pos", "full@1"]}
        if c["n_cl"]:
            r.update({k: round(100 * c[k] / c["n_cl"], 1) for k in ["contlex@1", "contlex@3"]})
        r["identity_lemma"] = round(100 * sum(f in {g[0] for g in map(tuple, json.loads(gd))} for f, gd in zip(forms.form, forms.gold)) / len(forms), 1)
        res[lang] = r
        print(lang, r, flush=True)
    res["strata"] = {f"{a}={b}": {k: round(100 * sc[k] / sc["n"], 1) for k in ["lemma@1", "lemma@5", "full@1"]}
                     | ({"contlex@1": round(100 * sc["contlex@1"] / sc["n_cl"], 1)} if sc["n_cl"] else {}) | {"n": sc["n"]}
                     for (a, b), sc in sorted(strata.items())}
    for k, x in res["strata"].items():
        print("  stratum", k, x, flush=True)
    res["_rows"] = rows_out
    for lang, df in ud.groupby("lang"):
        pred = predict(model, tok, list(zip(df.lang, df.form)))
        c = Counter(); j = 0
        for chunk in pred:
            for b in range(len(chunk["rule"][1])):
                row = df.iloc[j]; j += 1
                lemmas = [oov.apply_rule(row.form, v["rule"].itos[r]) if r else None for r in chunk["rule"][1][b].tolist()]
                c["n"] += 1
                c["lemma@1"] += (lemmas[0] or "").lower() == row.lemma.lower()
                top = lemmas[0] if chunk["rule"][0][b, 0].item() >= tau else row.form
                c["lemma@1_cal"] += (top or "").lower() == row.lemma.lower()
                c["lemma@5"] += any((l or "").lower() == row.lemma.lower() for l in lemmas + [row.form])
                c["upos"] += UPOS.get(v["pos"].itos[chunk["pos"][1][b, 0]]) == row.upos
                c["identity"] += row.form.lower() == row.lemma.lower()
        res[f"{lang}_ud_oov"] = {k: round(100 * c[k] / c["n"], 1) for k in ["lemma@1", "lemma@1_cal", "lemma@5", "upos", "identity"]} | {"n": c["n"]}
        print(lang, "UD-OOV", res[f"{lang}_ud_oov"], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args()
    L.seed_everything(1)
    torch.set_float32_matmul_precision("high")
    dfs = [pd.read_parquet(f"{W}/data/oov/{l}.parquet") for l in LANGS if os.path.exists(f"{W}/data/oov/{l}.parquet")]
    data = pd.concat(dfs, ignore_index=True)
    train = data[data.split == "train"]
    keep = train.drop_duplicates("form").groupby("lang").head(PER_LANG).form
    train = train[train.form.isin(set(keep))]
    test = data[data.split == "test"]
    test = test[test.form.isin(set(test.drop_duplicates("form").groupby("lang").head(3000).form))]
    uds = [pd.read_parquet(f"{W}/data/oov/{l}.ud_oov.parquet") for l in LANGS if os.path.exists(f"{W}/data/oov/{l}.ud_oov.parquet")]
    ud = pd.concat(uds, ignore_index=True) if uds else pd.DataFrame()
    v = {"rule": Vocab(train.rule), "pos": Vocab(train.pos, 1), "tags": Vocab(train.pos + "|" + train.tags),
         "contlex": Vocab((l + "|" + c for l, c in zip(train.lang, train.contlex) if c))}
    print({k: len(x.itos) for k, x in v.items()}, "train rows", len(train), flush=True)
    tok = AutoTokenizer.from_pretrained(a.encoder)
    rows = train.to_dict("records")
    dl = DataLoader(rows, batch_size=128, shuffle=True, collate_fn=lambda b: collate(b, tok, v), num_workers=4, drop_last=True)
    model = OOVModel(a.encoder, {k: len(x.itos) for k, x in v.items()}, steps=a.steps)
    L.Trainer(max_steps=a.steps, precision="bf16-mixed", enable_checkpointing=False, logger=False,
              enable_progress_bar=False).fit(model, dl)
    udt = [pd.read_parquet(f"{W}/data/oov/{l}.ud_oov_train.parquet") for l in LANGS if os.path.exists(f"{W}/data/oov/{l}.ud_oov_train.parquet")]
    tau = calibrate(model, tok, v, pd.concat(udt, ignore_index=True) if udt else pd.DataFrame())
    cl_counts = train.drop_duplicates(["lang", "lemma"])
    cl_freq = Counter(l + "|" + c for l, c in zip(cl_counts.lang, cl_counts.contlex) if c)
    res = evaluate(model, tok, v, test, ud, tau, cl_freq)
    res["tau"] = tau
    pd.DataFrame(res.pop("_rows")).to_parquet(f"{W}/runs/oov/{a.name}_predictions.parquet", index=False)
    os.makedirs(f"{W}/runs/oov", exist_ok=True)
    json.dump(res, open(f"{W}/runs/oov/{a.name}.json", "w"), indent=1)
    out = f"{W}/models/oov_{a.name}"
    os.makedirs(out, exist_ok=True)
    torch.save({"state": model.state_dict(), "vocabs": {k: x.itos for k, x in v.items()}, "encoder": a.encoder}, f"{out}/model.pt")


if __name__ == "__main__":
    main()
