"""morph_embeddings: static, CPU-friendly word vectors distilled from a (lexicon-grounded) encoder.

  v(w) = mean_{g in ngrams(w)} E_char[g]  +  mean_{a in A(w)} ( E_lemma[lang|lemma_a] + mean_{t in a} E_tag[t] )

A(w) are the FST analyses of w; for FST-unknown words, the unknown-word model's top hypothesis.
Teacher: the encoder's contextual vectors (mean of layers L1..L2, mean over sub-tokens), averaged over
up to MAX_OCC corpus occurrences per word type. Pivot languages (eng/fin/rus) are distilled into the same
space, so Uralic and pivot words share one vector space (used for BLI and English->Uralic transfer).

Stages:  teacher  -> work/emb/<name>/teacher_<lang>.pt
         train    -> work/emb/<name>/student.pt
         export   -> work/emb/<name>/<lang>.vec   (all teacher types + FST-known types)"""
import argparse, collections, json, os, pickle, re, sys, zlib
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

W = os.environ.get("LEXIBERT_WORK", "work")
URALIC = ["myv", "mdf", "kpv", "koi", "udm", "sme", "sms", "smn", "olo", "krl", "vep", "liv", "mhr", "mrj", "yrk"]
PIVOT = ["eng", "fin", "rus"]
CORE = re.compile(r"[^\W\d_]+(?:[’'ʼʹ-][^\W\d_]+)*")
MAX_SENTS, MAX_OCC, MAX_TYPES = int(os.environ.get("ME_SENTS", 300_000)), 64, int(os.environ.get("ME_TYPES", 200_000))
BUCKETS, DIM = 500_009, 384


def corpus_path(lang):
    return f"{W}/data/corpus_pivot/{lang}.txt" if lang in PIVOT else f"{W}/data/corpus/{lang}.txt"


@torch.no_grad()
def teacher(encoder, out, langs, layers):
    """Type-level teacher vectors, fully on the GPU: sub-token states -> word means (index_add) -> running per-type
    sums (index_add). Python only maps words to ids and enforces the MAX_OCC cap."""
    tok = AutoTokenizer.from_pretrained(encoder)
    enc = AutoModel.from_pretrained(encoder, dtype=torch.bfloat16).cuda().eval()
    enc.layers = enc.layers[:layers[1] + 1]  # one layer beyond the last averaged one keeps all used states identical (the last entry gets the final norm)
    dim = enc.config.hidden_size
    for lang in langs:
        path = corpus_path(lang)
        if not os.path.exists(path) or os.path.exists(f"{out}/teacher_{lang}.pt"):
            continue
        type_id, counts = {}, []
        sums = torch.zeros(1 << 16, dim, device="cuda")
        batch = []

        def flush():
            nonlocal sums
            words = [s.split()[:100] for s in batch]
            e = tok(words, is_split_into_words=True, truncation=True, max_length=192, padding=True, return_tensors="pt")
            hs = enc(input_ids=e.input_ids.cuda(), attention_mask=e.attention_mask.cuda(), output_hidden_states=True).hidden_states
            h = torch.stack(hs[layers[0]:layers[1] + 1]).float().mean(0)
            L = h.shape[1]
            tok_rows, tok_slot, slot_type = [], [], []
            for b, ws in enumerate(words):
                slot_of = {}
                for k, wi in enumerate(e.word_ids(b)):
                    if wi is None:
                        continue
                    if wi not in slot_of:
                        m = CORE.search(ws[wi])
                        if not m:
                            slot_of[wi] = -1
                            continue
                        w = m.group(0)
                        t = type_id.get(w)
                        if t is None:
                            t = type_id[w] = len(counts); counts.append(0)
                        if counts[t] >= MAX_OCC:
                            slot_of[wi] = -1
                            continue
                        counts[t] += 1
                        slot_of[wi] = len(slot_type); slot_type.append(t)
                    if slot_of[wi] >= 0:
                        tok_rows.append(b * L + k); tok_slot.append(slot_of[wi])
            if not slot_type:
                return
            if len(counts) > sums.shape[0]:
                grown = torch.zeros(max(len(counts), 2 * sums.shape[0]), dim, device="cuda")
                grown[:sums.shape[0]] = sums
                sums = grown
            rows = torch.tensor(tok_rows, device="cuda"); slots = torch.tensor(tok_slot, device="cuda")
            n = len(slot_type)
            word_sum = torch.zeros(n, dim, device="cuda").index_add_(0, slots, h.reshape(-1, dim)[rows])
            word_len = torch.zeros(n, device="cuda").index_add_(0, slots, torch.ones(len(tok_slot), device="cuda"))
            sums.index_add_(0, torch.tensor(slot_type, device="cuda"), word_sum / word_len[:, None])

        lines = [l.strip() for l, _ in zip(open(path, encoding="utf-8"), range(MAX_SENTS))]
        lines.sort(key=lambda l: len(l.split()))  # length bucketing: same sentences, far less padding
        for i in range(0, len(lines), 384):
            batch = lines[i:i + 384]
            flush()
        words_by_id = sorted(type_id, key=type_id.get)
        order = sorted(range(len(counts)), key=lambda t: -counts[t])[:MAX_TYPES]
        idx = torch.tensor(order, device="cuda")
        cnt = torch.tensor([counts[t] for t in order], device="cuda", dtype=torch.float)
        torch.save({"words": [words_by_id[t] for t in order], "vecs": (sums[idx] / cnt[:, None]).half().cpu(),
                    "counts": [counts[t] for t in order]}, f"{out}/teacher_{lang}.pt")
        print("teacher", lang, len(order), flush=True)


def ngrams(word, n_min=3, n_max=6):
    w = f"<{word}>"
    return [zlib.crc32(w[i:i + n].encode()) % BUCKETS for n in range(n_min, n_max + 1) for i in range(len(w) - n + 1)] or [0]


class Analyses:
    """FST analyses (cached) + unknown-word model hypotheses for FST-unknown words."""

    def __init__(self, langs, hyp_dir=None):
        self.a = {}
        for l in langs:
            p = f"{W}/data/fst/{l}.parquet"
            if os.path.exists(p):
                df = pd.read_parquet(p, columns=["word", "analyses"])
                self.a[l] = {w: list(a) for w, a in zip(df.word, df.analyses) if len(a)}
            # unknown-word hypotheses of one model; None = FST analyses only
            p = f"{hyp_dir}/{l}.parquet" if hyp_dir else ""
            if p and os.path.exists(p):
                df = pd.read_parquet(p)
                for w, a in zip(df.word, df.analysis):
                    self.a.setdefault(l, {}).setdefault(w, [a])

    def get(self, lang, word):
        d = self.a.get(lang, {})
        return d.get(word) or d.get(word.lower()) or []


def parse(analysis):
    analysis = analysis.split("#")[-1]
    parts = [p for p in analysis.split("+") if p]
    return (parts[0], parts[1:]) if parts else None


class Student(nn.Module):
    def __init__(self, n_lemmas, n_tags):
        super().__init__()
        self.char = nn.EmbeddingBag(BUCKETS, DIM, mode="mean", sparse=True)
        self.lemma = nn.Embedding(n_lemmas, DIM, sparse=True)
        self.tag = nn.EmbeddingBag(n_tags, DIM, mode="mean", sparse=True)
        for e in (self.char, self.lemma, self.tag):
            nn.init.normal_(e.weight, std=0.01)

    def forward(self, ng, ng_off, an_lemma, an_tags, an_tag_off, an_word, n_words):
        v = self.char(ng, ng_off)
        if len(an_lemma):
            a = self.lemma(an_lemma) + self.tag(an_tags, an_tag_off)
            agg = torch.zeros_like(v).index_add_(0, an_word, a)
            cnt = torch.zeros(n_words, device=v.device).index_add_(0, an_word, torch.ones(len(an_word), device=v.device))
            v = v + agg / cnt.clamp(min=1)[:, None]
        return v


class Featurizer:
    def __init__(self, analyses, lemma_idx, tag_idx, grow=False):
        self.an, self.li, self.ti, self.grow = analyses, lemma_idx, tag_idx, grow

    def lemma_id(self, key):
        if key not in self.li:
            if not self.grow:
                return None
            self.li[key] = len(self.li)
        return self.li[key]

    def __call__(self, items, device="cpu"):
        ng, ng_off, al, at, at_off, aw = [], [], [], [], [], []
        for i, (lang, w) in enumerate(items):
            ng_off.append(len(ng)); ng += ngrams(w)
            if lang in PIVOT:  # pivot words: whole-word "lemma" slot
                lid = self.lemma_id(f"{lang}|{w.lower()}")
                if lid is not None:
                    at_off.append(len(at)); at.append(self.ti["<none>"]); al.append(lid); aw.append(i)
                continue
            for a in self.an.get(lang, w)[:6]:
                p = parse(a)
                if not p:
                    continue
                lid = self.lemma_id(f"{lang}|{p[0]}")
                if lid is None:
                    continue
                at_off.append(len(at)); at += [self.ti.get(t, self.ti["<none>"]) for t in p[1]] or [self.ti["<none>"]]
                al.append(lid); aw.append(i)
        T = lambda x: torch.tensor(x, dtype=torch.long, device=device)
        return T(ng), T(ng_off), T(al), T(at), T(at_off), T(aw), len(items)


def train(out, langs, epochs, lr):
    tags = ["<none>"] + json.load(open(f"{W}/data/pretrain/tags.json"))
    tag_idx = {t: i for i, t in enumerate(tags)}
    hyp_dir = os.environ.get("ME_HYP") or None
    an = Analyses(langs, hyp_dir)
    data = []
    for lang in langs + PIVOT:
        p = f"{out}/teacher_{lang}.pt"
        if not os.path.exists(p):
            continue
        t = torch.load(p, map_location="cpu")
        for w, v, c in zip(t["words"], t["vecs"], t["counts"]):
            data.append((lang, w, v, c))
    print("distillation types", len(data), flush=True)
    lemma_idx = {}
    feat = Featurizer(an, lemma_idx, tag_idx, grow=True)
    for i in range(0, len(data), 4096):  # build the lemma vocabulary once
        feat([(l, w) for l, w, _, _ in data[i:i + 4096]])
    feat.grow = False
    model = Student(len(lemma_idx) + 1, len(tags)).cuda()
    opt = torch.optim.SparseAdam(list(model.parameters()), lr=lr)
    targets = F.normalize(torch.stack([d[2].float() for d in data]), dim=-1)
    weights = torch.tensor([np.log1p(d[3]) for d in data])
    for ep in range(epochs):
        perm = torch.randperm(len(data))
        tot = 0.0
        for i in range(0, len(data), 1024):
            idx = perm[i:i + 1024]
            x = feat([(data[j][0], data[j][1]) for j in idx.tolist()], "cuda")
            v = model(*x)
            loss = ((1 - F.cosine_similarity(v, targets[idx].cuda(), dim=-1)) * weights[idx].cuda()).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f"epoch {ep} loss {tot / len(data):.4f}", flush=True)
    torch.save({"state": model.state_dict(), "lemma_idx": lemma_idx, "tags": tags, "hyp_dir": hyp_dir}, f"{out}/student.pt")


def load_student(out, langs):
    ck = torch.load(f"{out}/student.pt", map_location="cpu")
    model = Student(len(ck["lemma_idx"]) + 1, len(ck["tags"]))
    model.load_state_dict(ck["state"])
    feat = Featurizer(Analyses(langs, ck.get("hyp_dir")), ck["lemma_idx"], {t: i for i, t in enumerate(ck["tags"])})
    return model.eval(), feat


@torch.no_grad()
def embed(model, feat, items, device="cpu"):
    model.to(device)
    return torch.cat([model(*feat(items[i:i + 4096], device)).cpu() for i in range(0, len(items), 4096)]) if items else torch.zeros(0, DIM)


def export(out, langs):
    model, feat = load_student(out, langs)
    for lang in langs + PIVOT:
        words = []
        p = f"{out}/teacher_{lang}.pt"
        if os.path.exists(p):
            words += torch.load(p, map_location="cpu")["words"]
        if lang in feat.an.a:
            words += list(feat.an.a[lang].keys())[:MAX_TYPES]
        words = list(dict.fromkeys(words))
        if not words:
            continue
        vecs = embed(model, feat, [(lang, w) for w in words], "cuda")
        with open(f"{out}/{lang}.vec", "w", encoding="utf-8") as f:
            f.write(f"{len(words)} {DIM}\n")
            for w, v in zip(words, vecs.numpy()):
                f.write(w + " " + " ".join(f"{x:.4f}" for x in v) + "\n")
        print("export", lang, len(words), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--stages", nargs="+", default=["teacher", "train", "export"])
    ap.add_argument("--layers", nargs=2, type=int, default=[8, 16])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--langs", nargs="+", default=URALIC)
    a = ap.parse_args()
    out = f"{W}/emb/{a.name}"
    os.makedirs(out, exist_ok=True)
    if "teacher" in a.stages:
        teacher(a.encoder, out, a.langs + PIVOT, a.layers)
    if "train" in a.stages:
        train(out, a.langs, a.epochs, a.lr)
    if "export" in a.stages:
        export(out, a.langs)


if __name__ == "__main__":
    main()
