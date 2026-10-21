"""Static-vector UD benchmark: identical BiLSTM tagger + biaffine parser, only the frozen word vectors change.

Architecture (Stanza-like): [frozen pretrained vector ; trainable word emb (freq>=2) ; char-BiLSTM]
 -> 2-layer BiLSTM -> UPOS / UFeats / lemma-rule heads + biaffine arcs and labels.
Vector sources: none | stanza (surface lookup) | a2023 (surface, then FST-lemma lookup) | fasttext (n-gram
composition for OOV) | morph:<emb dir> (FST + char composition for every word, incl. unknown)."""
import argparse, json, os, sys
from collections import Counter
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
ud = import_module("07_ud")

W = os.environ.get("LEXIBERT_WORK", "work")
TB_LANG = {"myv": "myv", "mdf": "mdf", "kpv_lattice": "kpv", "kpv_ikdp": "kpv", "koi": "koi", "sms": "sms", "olo": "olo",
           "krl": "krl", "sme": "sme", "vep": "vep", "yrk": "yrk", "fin": "fin", "est": "est"}
STANZA = {"myv": "myv/pretrain/rootroo.pt", "sme": "sme/pretrain/fasttextwiki.pt", "kpv": "kpv/pretrain/fasttextwiki.pt",
          "fin": "fi/pretrain/conll17.pt", "est": "et/pretrain/conll17.pt"}


def read_vec(path, limit=None):
    vecs = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        n, d = map(int, f.readline().split())
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            p = line.rstrip().split(" ")
            if len(p) == d + 1:
                vecs[p[0]] = np.asarray(p[1:], dtype=np.float32)
    return vecs, d


class Source:
    """Maps (lang, word) -> vector. Every source is L2-normalised to make scales comparable."""

    def __init__(self, kind, langs):
        self.kind, self.cache, self.dim = kind, {}, 1
        self.tables, self.models = {}, {}
        if kind == "none":
            return
        if kind == "stanza":
            from stanza.models.common.pretrain import Pretrain
            for l in langs:
                if l in STANZA:
                    p = Pretrain(f"{W}/stanza/{STANZA[l]}")
                    self.tables[l] = {w: p.emb[i].numpy() for i, w in enumerate(p.vocab._id2unit) if i > 3}
                    self.dim = p.emb.shape[1]
        elif kind == "a2023":
            for l in langs:
                path = f"{W}/downloads/zenodo_7866456/x/vectors/vectors-{l}.txt"
                if os.path.exists(path):
                    self.tables[l], self.dim = read_vec(path)
        elif kind == "fasttext":
            from gensim.models import FastText
            for l in langs:
                path = f"{W}/emb/fasttext/{l}.model"
                if os.path.exists(path):
                    self.models[l] = FastText.load(path).wv
                    self.dim = self.models[l].vector_size
        elif kind.startswith("morph:"):
            me = import_module("11_morph_embeddings")
            self.me = me
            self.student, self.feat = me.load_student(f"{W}/emb/{kind.split(':', 1)[1]}", sorted(set(langs)))
            self.dim = me.DIM

    def get(self, lang, word):
        key = (lang, word)
        if key in self.cache:
            return self.cache[key]
        v = None
        if self.kind in ("stanza", "a2023") and lang in self.tables:
            t = self.tables[lang]
            v = t.get(word, t.get(word.lower()))
            if v is None and self.kind == "a2023":  # vectors were trained on lemmatised text: look up FST lemmas
                lem = [t[l] for l in ud.fst_lemmas(next((k for k, x in TB_LANG.items() if x == lang), lang), word) if l in t]
                v = np.mean(lem, 0) if lem else None
        elif self.kind == "fasttext" and lang in self.models:
            v = self.models[lang][word]
        elif self.kind.startswith("morph:"):
            v = self.me.embed(self.student, self.feat, [(lang, word)])[0].numpy()
        if v is not None:
            v = v / (np.linalg.norm(v) + 1e-8)
        self.cache[key] = v
        return v

    def coverage(self, sents, lang):
        found = sum(self.get(lang, t["form"]) is not None for s in sents for t in s)
        return round(100 * found / max(1, sum(len(s) for s in sents)), 1)


class Tagger(L.LightningModule):
    def __init__(self, n_words, n_chars, dim_pre, sizes, steps):
        super().__init__()
        self.save_hyperparameters()
        self.wemb = nn.Embedding(n_words, 100, padding_idx=0)
        self.cemb = nn.Embedding(n_chars, 64, padding_idx=0)
        self.clstm = nn.LSTM(64, 100, batch_first=True, bidirectional=True)
        self.pre_proj = nn.Linear(dim_pre, 125)
        self.lstm = nn.LSTM(100 + 200 + 125, 400, num_layers=2, batch_first=True, bidirectional=True, dropout=0.33)
        self.drop = nn.Dropout(0.33)
        H = 800
        self.upos, self.feats, self.lemma = nn.Linear(H, sizes["upos"]), nn.Linear(H, sizes["feats"]), nn.Linear(H, sizes["lemma"])
        self.arc_h, self.arc_d = nn.Linear(H, 400), nn.Linear(H, 400)
        self.arc_W = nn.Parameter(torch.zeros(400, 400)); self.arc_b = nn.Linear(400, 1, bias=False)
        self.rel_h, self.rel_d = nn.Linear(H, 100), nn.Linear(H, 100)
        self.rel_W = nn.Parameter(torch.zeros(sizes["rel"], 101, 101))
        self.root = nn.Parameter(torch.randn(H) * 0.02)

    def forward(self, words, chars, pre, wmask):
        B, T, C = chars.shape
        ch = self.cemb(chars.view(B * T, C))
        _, (h, _) = self.clstm(ch)
        ch = torch.cat([h[0], h[1]], -1).view(B, T, -1)
        x = torch.cat([self.wemb(words), ch, self.pre_proj(pre)], -1)
        x, _ = self.lstm(self.drop(x))
        x = self.drop(x)
        w = torch.cat([self.root.expand(B, 1, -1), x], 1)
        ah, ad = F.gelu(self.arc_h(w)), F.gelu(self.arc_d(w))
        arc = torch.einsum("bie,ef,bjf->bij", ad, self.arc_W, ah) + self.arc_b(ah).transpose(1, 2)
        full = torch.cat([torch.ones_like(wmask[:, :1]), wmask], 1)
        arc = arc.masked_fill(~full[:, None, :], -1e4)
        rh = torch.cat([F.gelu(self.rel_h(w)), torch.ones_like(w[..., :1])], -1)
        rd = torch.cat([F.gelu(self.rel_d(w)), torch.ones_like(w[..., :1])], -1)
        return self.upos(x), self.feats(x), self.lemma(x), arc[:, 1:], rh, rd[:, 1:]

    def rel_scores(self, rh, rd, heads):
        return torch.einsum("bie,ref,bif->bir", rd, self.rel_W, rh[torch.arange(rh.size(0))[:, None], heads])

    def training_step(self, batch, _):
        words, chars, pre, m, upos, feats, lemma, head, rel = batch
        pu, pf, pl, arc, rh, rd = self(words, chars, pre, m)
        loss = sum(F.cross_entropy(p[m], y[m]) for p, y in [(pu, upos), (pf, feats), (pl, lemma), (arc, head)])
        loss = loss + F.cross_entropy(self.rel_scores(rh, rd, head)[m], rel[m])
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=2e-3, betas=(0.9, 0.9), weight_decay=1e-5)
        s = self.hparams.steps
        return [opt], [{"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lambda i: max(0.05, 1 - i / s)), "interval": "step"}]


def build(sents_by_tb, src, wv, cv, v):
    items = []
    for tb, sents in sents_by_tb.items():
        lang = TB_LANG[tb]
        for s in sents:
            pre = np.stack([src.get(lang, t["form"]) if src.get(lang, t["form"]) is not None else np.zeros(src.dim, np.float32) for t in s])
            items.append(([wv(t["form"].lower()) for t in s], [[cv(c) for c in t["form"][:20]] or [0] for t in s], pre,
                          [v["upos"](t["upos"]) for t in s], [v["feats"](t["feats"]) for t in s],
                          [v["lemma"](ud.lemma_rule(t["form"], t["lemma"])) for t in s], [t["head"] for t in s], [v["rel"](t["rel"]) for t in s]))
    return items


def collate(items):
    B, T = len(items), max(len(i[0]) for i in items)
    C = max(len(c) for i in items for c in i[1])
    words = torch.zeros(B, T, dtype=torch.long); chars = torch.zeros(B, T, C, dtype=torch.long)
    pre = torch.zeros(B, T, items[0][2].shape[1]); m = torch.zeros(B, T, dtype=torch.bool)
    lab = [torch.zeros(B, T, dtype=torch.long) for _ in range(5)]
    for b, it in enumerate(items):
        n = len(it[0])
        words[b, :n] = torch.tensor(it[0]); pre[b, :n] = torch.from_numpy(it[2]); m[b, :n] = True
        for j, c in enumerate(it[1]):
            chars[b, j, :len(c)] = torch.tensor(c)
        for k in range(5):
            lab[k][b, :n] = torch.tensor(it[3 + k])
    return [words, chars, pre, m] + lab


@torch.no_grad()
def evaluate(model, sents, items, v, tb):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(dev)
    c = Counter()
    for i in range(0, len(items), 64):
        batch = [x.to(dev) for x in collate(items[i:i + 64])]
        pu, pf, pl, arc, rh, rd = model(*batch[:4])
        heads = ud.mst_heads(arc, batch[3]).to(arc.device); rels = model.rel_scores(rh, rd, heads).argmax(-1)
        pu, pf, pl, heads, rels, pll = [x.cpu() for x in (pu.argmax(-1), pf.argmax(-1), pl.argmax(-1), heads, rels, pl.float())]
        for b, s in enumerate(sents[i:i + 64]):
            for j, t in enumerate(s):
                c["n"] += 1
                c["upos"] += v["upos"].itos[pu[b, j]] == t["upos"]
                c["feats"] += v["feats"].itos[pf[b, j]] == t["feats"]
                pred = ud.apply_rule(t["form"], v["lemma"].itos[pl[b, j]]) if pl[b, j] else t["form"]
                c["lemma"] += pred.lower() == t["lemma"].lower()
                pred = ud.fst_lemma(t["form"], pred, ud.fst_lemmas(tb, t["form"]), pll[b, j], v)
                c["lemma_fst"] += pred.lower() == t["lemma"].lower()
                ok = int(heads[b, j]) == t["head"]
                c["uas"] += ok; c["las"] += ok and v["rel"].itos[rels[b, j]] == t["rel"]
    return {k: round(100 * c[k] / c["n"], 2) for k in ["upos", "feats", "lemma", "lemma_fst", "uas", "las"]}


RUNS = os.environ.get("RUNS", "runs")  # results dir under work/ (reruns write elsewhere so they can be compared)
CKPT = f"{W}/checkpoints"


def main():
    torch.set_num_threads(int(os.environ.get("STATIC_THREADS", 8)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--setting", choices=["mono", "multi"], required=True)
    ap.add_argument("--train", nargs="+", default=["myv"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    L.seed_everything(a.seed)
    if a.setting == "mono":
        train = {a.train[0]: ud.split(a.train[0], "train")}
        tests = [a.train[0]]
    else:
        import random
        rng = random.Random(a.seed)
        train = {tb: ud.split(tb, "train") for tb in ud.TB}
        for tb, cap in ud.SOURCE_CAP.items():
            rng.shuffle(train[tb]); train[tb] = train[tb][:cap]
        tests = [tb for tb in ud.TB if tb not in ud.SOURCE_CAP]
    all_train = [s for x in train.values() for s in x]
    v = {"upos": ud.Vocab(t["upos"] for s in all_train for t in s), "feats": ud.Vocab(t["feats"] for s in all_train for t in s),
         "lemma": ud.Vocab((ud.lemma_rule(t["form"], t["lemma"]) for s in all_train for t in s), min_count=2),
         "rel": ud.Vocab(t["rel"] for s in all_train for t in s)}
    wv = ud.Vocab((t["form"].lower() for s in all_train for t in s), min_count=2)
    cv = ud.Vocab((c for s in all_train for t in s for c in t["form"]), min_count=1)
    src = Source(a.vectors, [TB_LANG[tb] for tb in list(train) + tests])
    items = build(train, src, wv, cv, v)
    dl = DataLoader(items, batch_size=32, shuffle=True, collate_fn=collate, drop_last=True)
    model = Tagger(len(wv.itos), len(cv.itos), src.dim, {k: len(x.itos) for k, x in v.items()}, a.steps)
    L.Trainer(max_steps=a.steps, gradient_clip_val=5.0, enable_checkpointing=False, logger=False,
              enable_progress_bar=False).fit(model, dl)
    os.makedirs(f"{CKPT}/ud_static", exist_ok=True)
    torch.save({"state": model.state_dict(), "vocabs": {k: x.itos for k, x in v.items()}, "words": wv.itos, "chars": cv.itos,
                "args": vars(a)}, f"{CKPT}/ud_static/{a.name}_{a.setting}_s{a.seed}.pt")
    res = {}
    for tb in tests:
        ts = ud.split(tb, "test")
        res[tb] = evaluate(model, ts, build({tb: ts}, src, wv, cv, v), v, tb) | {"vec_coverage": src.coverage(ts, TB_LANG[tb])}
        print(tb, res[tb], flush=True)
    os.makedirs(f"{W}/{RUNS}/ud_static", exist_ok=True)
    json.dump(dict(args=vars(a), results=res), open(f"{W}/{RUNS}/ud_static/{a.name}_{a.setting}_s{a.seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
