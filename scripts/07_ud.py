"""UD evaluation harness: encoder + UPOS / UFeats / lemma-rule heads + biaffine parser.

Settings
  mono   train on one treebank's train split, test on its test split (myv, sme)
  multi  one model trained on all Uralic train splits (+ fin/est), tested on every test set
Metrics on gold tokenisation: UPOS, UFeats, Lemma, UAS, LAS (punctuation included, CoNLL-18 style)."""
import argparse, glob, json, os, random
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

W = os.environ.get("LEXIBERT_WORK", "work")
UD = f"{W}/data/ud"
TB = {"myv": "UD_Erzya-JR", "mdf": "UD_Moksha-JR", "kpv_lattice": "UD_Komi_Zyrian-Lattice", "kpv_ikdp": "UD_Komi_Zyrian-IKDP",
      "koi": "UD_Komi_Permyak-UH", "sms": "UD_Skolt_Sami-Giellagas", "olo": "UD_Livvi-KKPP", "krl": "UD_Karelian-KKPP",
      "sme": "UD_North_Sami-Giella", "vep": "UD_Veps-VWT", "yrk": "UD_Nenets-Tundra", "fin": "UD_Finnish-TDT", "est": "UD_Estonian-EDT"}
SOURCE_CAP = {"fin": 6000, "est": 4000}
FST_LANG = {"myv": "myv", "mdf": "mdf", "kpv_lattice": "kpv", "kpv_ikdp": "kpv", "koi": "koi", "sms": "sms",
            "olo": "olo", "krl": "krl", "sme": "sme", "vep": "vep", "yrk": "yrk"}
_FST = {}


def fst_lemmas(tb, form):
    """Candidate lemmas for a form from the cached FST analyses (compounds/derivations skipped)."""
    lang = FST_LANG.get(tb)
    if lang is None:
        return set()
    if lang not in _FST:
        import pandas as pd
        path = f"{W}/data/fst/{lang}.parquet"
        df = pd.read_parquet(path) if os.path.exists(path) else None
        _FST[lang] = {} if df is None else dict(zip(df.word, df.analyses))
    cands = set()
    for a in list(_FST[lang].get(form, [])) or list(_FST[lang].get(form.lower(), [])):
        if "#" not in a:
            cands.add(a.split("+")[0])
    return cands


def mst_heads(arc, wmask):
    """Heads as the highest scoring tree with exactly one root (Chu-Liu-Edmonds, as in Stanza).
    arc: (B, n, n+1) scores of each word (rows) for each head (columns, 0 = ROOT); wmask: (B, n) words that were encoded."""
    import numpy as np
    from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root
    arc, wmask = arc.float().cpu(), wmask.cpu()
    heads = torch.zeros(wmask.shape, dtype=torch.long)
    for b in range(arc.size(0)):
        k = int(wmask[b].sum())  # truncation only drops words at the end, so the encoded words are a prefix
        if k == 0:
            continue
        sc = arc[b, :k, :k + 1].clone()
        sc[torch.arange(k), torch.arange(1, k + 1)] = -float("inf")  # no self-loops
        scores = np.zeros((k + 1, k + 1))
        scores[1:] = F.log_softmax(sc, -1).numpy()
        heads[b, :k] = torch.from_numpy(chuliu_edmonds_one_root(scores)[1:].astype(np.int64))
    return heads


def fst_lemma(form, pred, cands, logits, v):
    """Restrict the predicted lemma to the FST lemmas: the one whose rule the lemma head scores highest, otherwise the
    shortest one (ties broken alphabetically, so that the result does not depend on the order of the set)."""
    if not cands or pred in cands:
        return pred
    scored = [(logits[v["lemma"](lemma_rule(form, c))].item(), c) for c in cands if v["lemma"](lemma_rule(form, c))]
    return max(scored)[1] if scored else min(cands, key=lambda c: (len(c), c))


def read(path):
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if cur:
                sents.append(cur)
            cur = []
            continue
        if line.startswith("#"):
            continue
        p = line.split("\t")
        if not p[0].isdigit():
            continue
        cur.append(dict(form=p[1], lemma=p[2], upos=p[3], feats=p[5], head=int(p[6]) if p[6].isdigit() else 0, rel=p[7].split(":")[0]))
    if cur:
        sents.append(cur)
    return sents


def split(tb, name):
    f = glob.glob(f"{UD}/{TB[tb]}/*-ud-{name}.conllu")
    return read(f[0]) if f else []


def lemma_rule(form, lemma):
    f, l = form.lower(), lemma.lower()
    p = 0
    while p < min(len(f), len(l)) and f[p] == l[p]:
        p += 1
    case = "U" if lemma[:1].isupper() else "L"
    return f"{case}|{len(f) - p}|{l[p:]}"


def apply_rule(form, rule):
    case, k, s = rule.split("|", 2)
    k = int(k)
    if k > len(form):
        return form
    out = (form.lower()[:-k] if k else form.lower()) + s
    return out[:1].upper() + out[1:] if case == "U" else out


class Vocab:
    def __init__(self, items, min_count=1):
        c = Counter(items)
        self.itos = ["<unk>"] + [x for x, n in c.most_common() if n >= min_count]
        self.stoi = {x: i for i, x in enumerate(self.itos)}

    def __call__(self, x):
        return self.stoi.get(x, 0)


class Parser(L.LightningModule):
    def __init__(self, encoder, vocabs, lr=5e-5, head_lr=1e-3, steps=3000):
        super().__init__()
        self.save_hyperparameters(ignore=["vocabs"])
        self.enc = AutoModel.from_pretrained(encoder, attn_implementation="sdpa") if "bert" in encoder.lower() or "mmBERT" in encoder else AutoModel.from_pretrained(encoder)
        H = self.enc.config.hidden_size
        self.v = vocabs
        self.mix = nn.Parameter(torch.zeros(self.enc.config.num_hidden_layers + 1))
        self.drop = nn.Dropout(0.2)
        self.upos = nn.Linear(H, len(vocabs["upos"].itos))
        self.feats = nn.Linear(H, len(vocabs["feats"].itos))
        self.lemma = nn.Linear(H, len(vocabs["lemma"].itos))
        self.arc_h, self.arc_d = nn.Linear(H, 384), nn.Linear(H, 384)
        self.rel_h, self.rel_d = nn.Linear(H, 128), nn.Linear(H, 128)
        self.arc_W = nn.Parameter(torch.zeros(384, 384)); self.arc_b = nn.Linear(384, 1, bias=False)
        self.rel_W = nn.Parameter(torch.zeros(len(vocabs["rel"].itos), 129, 129))
        self.root = nn.Parameter(torch.randn(H) * 0.02)

    def words(self, ids, att, first):
        out = self.enc(input_ids=ids, attention_mask=att, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 0)
        h = (F.softmax(self.mix, 0)[:, None, None, None] * hs).sum(0)
        B, Wn = first.shape
        w = h[torch.arange(B)[:, None], first.clamp(min=0)]
        w = torch.cat([self.root.expand(B, 1, -1), w], 1)  # index 0 = ROOT
        return self.drop(w)

    def forward(self, ids, att, first, wmask):
        w = self.words(ids, att, first)
        tok = w[:, 1:]
        ah, ad = F.gelu(self.arc_h(w)), F.gelu(self.arc_d(w))
        arc = torch.einsum("bie,ef,bjf->bij", ad, self.arc_W, ah) + self.arc_b(ah).transpose(1, 2)
        full = torch.cat([torch.ones_like(wmask[:, :1]), wmask], 1)
        arc = arc.masked_fill(~full[:, None, :], -1e4)
        rh = torch.cat([F.gelu(self.rel_h(w)), torch.ones_like(w[..., :1])], -1)
        rd = torch.cat([F.gelu(self.rel_d(w)), torch.ones_like(w[..., :1])], -1)
        return self.upos(tok), self.feats(tok), self.lemma(tok), arc[:, 1:], rh, rd[:, 1:]

    def rel_scores(self, rh, rd, heads):
        hsel = rh[torch.arange(rh.size(0))[:, None], heads]
        return torch.einsum("bie,ref,bif->bir", rd, self.rel_W, hsel)

    def step(self, batch):
        ids, att, first, wmask, upos, feats, lemma, head, rel = batch
        pu, pf, pl, arc, rh, rd = self(ids, att, first, wmask)
        m = wmask
        loss = F.cross_entropy(pu[m], upos[m]) + F.cross_entropy(pf[m], feats[m]) + F.cross_entropy(pl[m], lemma[m])
        loss = loss + F.cross_entropy(arc[m], head[m]) + F.cross_entropy(self.rel_scores(rh, rd, head)[m], rel[m])
        return loss

    def training_step(self, batch, _):
        loss = self.step(batch)
        self.log("loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def predict(self, batch):
        ids, att, first, wmask = batch[:4]
        pu, pf, pl, arc, rh, rd = self(ids, att, first, wmask)
        heads = mst_heads(arc, wmask).to(arc.device)
        rels = self.rel_scores(rh, rd, heads).argmax(-1)
        return pu.argmax(-1), pf.argmax(-1), pl.argmax(-1), heads, rels

    def configure_optimizers(self):
        enc = [p for n, p in self.named_parameters() if n.startswith("enc.")]
        rest = [p for n, p in self.named_parameters() if not n.startswith("enc.")]
        opt = torch.optim.AdamW([{"params": enc, "lr": self.hparams.lr}, {"params": rest, "lr": self.hparams.head_lr}], weight_decay=0.01)
        s = self.hparams.steps
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / (0.1 * s)) * max(0.0, (s - i) / s))
        return [opt], [{"scheduler": sched, "interval": "step"}]


def encode(sents, tok, v, maxlen=256):
    items = []
    for s in sents:
        enc = tok([t["form"] for t in s], is_split_into_words=True, truncation=True, max_length=maxlen)
        first, seen = [-1] * len(s), set()
        for k, wi in enumerate(enc.word_ids()):
            if wi is not None and wi not in seen:
                seen.add(wi); first[wi] = k
        items.append((enc.input_ids, first, [v["upos"](t["upos"]) for t in s], [v["feats"](t["feats"]) for t in s],
                      [v["lemma"](lemma_rule(t["form"], t["lemma"])) for t in s], [t["head"] for t in s], [v["rel"](t["rel"]) for t in s]))
    return items


def collate(items, pad):
    B, Lt, Lw = len(items), max(len(i[0]) for i in items), max(len(i[1]) for i in items)
    ids = torch.full((B, Lt), pad); att = torch.zeros(B, Lt, dtype=torch.long)
    first = torch.zeros(B, Lw, dtype=torch.long); wmask = torch.zeros(B, Lw, dtype=torch.bool)
    lab = [torch.zeros(B, Lw, dtype=torch.long) for _ in range(5)]
    for b, it in enumerate(items):
        ids[b, :len(it[0])] = torch.tensor(it[0]); att[b, :len(it[0])] = 1
        fw = torch.tensor(it[1]); first[b, :len(fw)] = fw.clamp(min=0); wmask[b, :len(fw)] = fw >= 0
        for j in range(5):
            lab[j][b, :len(it[2 + j])] = torch.tensor(it[2 + j])
    lab[3] = lab[3].clamp(max=Lw)
    return [ids, att, first, wmask] + lab


DUMP = [] if os.environ.get("UD_DUMP") else None
RUNS = os.environ.get("RUNS", "runs")  # results dir under work/ (reruns write elsewhere so they can be compared)
CKPT = f"{W}/checkpoints"


@torch.no_grad()
def evaluate(model, sents, items, v, pad, device, tb=None):
    model.eval().to(device)
    c = Counter()
    for i in range(0, len(items), 32):
        batch = [x.to(device) for x in collate(items[i:i + 32], pad)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pu, pf, pl, ph, pr = [x.cpu() for x in model.predict(batch)]
            lemma_logits = model.lemma(model.words(*batch[:3])[:, 1:]).float().cpu()
        for b, s in enumerate(sents[i:i + 32]):
            for j, t in enumerate(s):
                if j >= len(items[i + b][1]) or items[i + b][1][j] < 0:
                    c["n"] += 1
                    continue
                c["n"] += 1
                c["upos"] += v["upos"].itos[pu[b, j]] == t["upos"]
                c["feats"] += v["feats"].itos[pf[b, j]] == t["feats"]
                c["lemma"] += apply_rule(t["form"], v["lemma"].itos[pl[b, j]]).lower() == t["lemma"].lower() if pl[b, j] else t["form"].lower() == t["lemma"].lower()
                pred = apply_rule(t["form"], v["lemma"].itos[pl[b, j]]) if pl[b, j] else t["form"]
                cands = fst_lemmas(tb, t["form"])
                pred = fst_lemma(t["form"], pred, cands, lemma_logits[b, j], v)
                c["lemma_fst"] += pred.lower() == t["lemma"].lower()
                if DUMP is not None and cands:  # examples for the paper: encoder lemma vs FST-restricted lemma
                    raw = apply_rule(t["form"], v["lemma"].itos[pl[b, j]]) if pl[b, j] else t["form"]
                    DUMP.append(dict(tb=tb, sent=" ".join(x["form"] for x in s), form=t["form"], gold=t["lemma"],
                                     enc=raw, fst=pred, cands=sorted(cands)))
                c["fst_known"] += bool(cands)
                uas = int(ph[b, j]) == t["head"]
                c["uas"] += uas
                c["las"] += uas and v["rel"].itos[pr[b, j]] == t["rel"]
    return {k: round(100 * c[k] / c["n"], 2) for k in ["upos", "feats", "lemma", "lemma_fst", "uas", "las", "fst_known"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--setting", choices=["mono", "multi"], required=True)
    ap.add_argument("--train", nargs="+", default=["myv"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    L.seed_everything(a.seed)
    torch.set_float32_matmul_precision("high")
    rng = random.Random(a.seed)
    if a.setting == "mono":
        train = {a.train[0]: split(a.train[0], "train")}
        tests = [a.train[0]]
    else:
        train = {tb: split(tb, "train") for tb in TB}
        for tb, cap in SOURCE_CAP.items():
            rng.shuffle(train[tb]); train[tb] = train[tb][:cap]
        tests = [tb for tb in TB if tb not in SOURCE_CAP]
    all_train = [s for v in train.values() for s in v]
    v = {"upos": Vocab(t["upos"] for s in all_train for t in s), "feats": Vocab(t["feats"] for s in all_train for t in s),
         "lemma": Vocab((lemma_rule(t["form"], t["lemma"]) for s in all_train for t in s), min_count=2),
         "rel": Vocab(t["rel"] for s in all_train for t in s)}
    tok = AutoTokenizer.from_pretrained(a.encoder)
    items = encode(all_train, tok, v)
    dl = DataLoader(items, batch_size=16, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id), drop_last=True)
    model = Parser(a.encoder, v, steps=a.steps)
    trainer = L.Trainer(max_steps=a.steps, precision="bf16-mixed", gradient_clip_val=5.0, enable_checkpointing=False,
                        logger=False, enable_progress_bar=False)
    trainer.fit(model, dl)
    os.makedirs(f"{CKPT}/ud", exist_ok=True)
    torch.save({"state": model.state_dict(), "vocabs": {k: x.itos for k, x in v.items()}, "args": vars(a)},
               f"{CKPT}/ud/{a.name}_{a.setting}_s{a.seed}.pt")
    res = {}
    for tb in tests:
        ts = split(tb, "test")
        res[tb] = evaluate(model, ts, encode(ts, tok, v), v, tok.pad_token_id, "cuda", tb)
        print(tb, res[tb], flush=True)
    if DUMP is not None:
        json.dump(DUMP, open(os.environ["UD_DUMP"], "w"), ensure_ascii=False)
        return
    os.makedirs(f"{W}/{RUNS}/ud", exist_ok=True)
    json.dump(dict(args=vars(a), results=res), open(f"{W}/{RUNS}/ud/{a.name}_{a.setting}_s{a.seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
