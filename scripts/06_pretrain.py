"""Lexicon-grounded continued pretraining of mmBERT-small.

Recipe follows mmBERT's annealed language learning: the 16 Uralic languages are mixed with English, Finnish and
Russian replay text; masking anneals 30% -> 15% -> 5% and the sampling temperature 0.7 -> 0.5 -> 0.3 (p ~ n^tau),
with phases in mmBERT's proportions (77% / 20% / 3% of the steps).

Objectives (switch on with flags, equal compute for every variant):
  MLM                    always (annealed masking, sparse prediction)
  --fst                  word-level soft multi-label FST-tag head on the first sub-token of each word
  --cs                   dictionary code-switching: 15% of words -> Ve'rdd translation (eng/fin/rus/est)
  --par                  translation-LM on Ve'rdd example pairs (mhr-rus, kpv-rus), 10% of batches
  --gloss                translation-LM on dictionary glosses: 25% of Uralic samples become "sentence [SEP] English
                         word-by-word gloss" (27_prep_gloss), with MLM over both
  --align                gloss alignment: for 25% of Uralic samples, every glossed word (first sub-token) and the
                         mean of the sentence are pulled towards the representations of the English gloss, which is
                         encoded by the same model without gradient (an English anchor)
"""
import argparse, glob, json, multiprocessing, os, pickle, random, re
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import lightning as L
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

W = os.environ.get("LEXIBERT_WORK", "work")
P = f"{W}/data/pretrain"
BASE = "jhu-clsp/mmBERT-small"
LANGS = ["myv", "mdf", "kpv", "koi", "udm", "sme", "sms", "smn", "olo", "krl", "vep", "liv", "mhr", "mrj", "yrk", "mns"]
PIVOTS = ["eng", "fin", "rus"]  # high-resource replay, as mmBERT never trains without them
PHASES = [(0.77, 0.30, 0.7), (0.97, 0.15, 0.5), (1.01, 0.05, 0.3)]  # (end fraction, mask rate, temperature)
STEP = multiprocessing.Value("i", 0)  # global step, shared with the data-loader workers (fork)
TOTAL = multiprocessing.Value("i", 1)


def phase():
    frac = STEP.value / max(1, TOTAL.value)
    return next(p for p in PHASES if frac < p[0])
CORE = re.compile(r"[^\W\d_]+(?:[’'ʼʹ-][^\W\d_]+)*")
MAXLEN, WORDS_PER_SAMPLE = 128, 40


class Stream(IterableDataset):
    def __init__(self, args, tok, n_tags):
        self.a, self.tok, self.n_tags = args, tok, n_tags

    def _setup(self):
        rng = random.Random(1000 + (get_worker_info().id if get_worker_info() else 0))
        self.rng = rng
        self.pool = LANGS + PIVOTS
        self.sents = {l: open(f"{W}/data/corpus/{l}.txt", encoding="utf-8").read().splitlines() for l in LANGS}
        self.sents.update({l: open(f"{W}/data/corpus_pivot/{l}.txt", encoding="utf-8").read().splitlines() for l in PIVOTS})
        self.sizes = np.array([len(self.sents[l]) for l in self.pool], dtype=float)
        self.tau, self.p = None, None
        self.fst = {l: pickle.load(open(f"{P}/fst_{l}.pkl", "rb")) for l in LANGS} if self.a.fst else {}
        self.cs = pickle.load(open(f"{P}/codeswitch.pkl", "rb")) if self.a.cs else {}
        self.par = json.load(open(f"{P}/parallel_train.json")) if self.a.par else []
        self.gloss = {l: pickle.load(open(f"{P}/gloss_{l}.pkl", "rb")) for l in LANGS
                      if os.path.exists(f"{P}/gloss_{l}.pkl")} if (self.a.gloss or self.a.align) else {}

    def _sample_words(self):
        tau = phase()[2]
        if tau != self.tau:
            p = self.sizes ** tau
            self.tau, self.p = tau, p / p.sum()
        lang = self.pool[self.rng.choices(range(len(self.pool)), weights=self.p)[0]]
        s = self.sents[lang]
        i = self.rng.randrange(len(s))
        words = []
        while len(words) < WORDS_PER_SAMPLE and i < len(s):
            words += s[i].split()
            i += 1
        return lang, words[:WORDS_PER_SAMPLE]

    def __iter__(self):
        self._setup()
        while True:
            if self.par and self.rng.random() < 0.1:
                src, a, b = self.rng.choice(self.par)
                words, lang, segs = a.split() + [self.tok.sep_token] + b.split(), src, None
            else:
                lang, words = self._sample_words()
            glossed = None
            if lang in self.gloss and self.rng.random() < 0.25:
                glossed = []
                for w in words:
                    m = CORE.search(w)
                    t = self.gloss[lang].get(m.group(0)) if m else None
                    glossed.append(t if t else None)
                if self.a.gloss:  # translation-LM: the sentence and its gloss as one input
                    words = words[:20] + [self.tok.sep_token] + [g or w for w, g in zip(words[:20], glossed[:20])]
                    glossed = None
            targets, switched = [], []
            for j, w in enumerate(words):
                m = CORE.search(w)
                core = m.group(0) if m else None
                tgt = self.fst.get(lang, {}).get(core) if core else None
                if self.cs and core and self.rng.random() < 0.15 and core in self.cs.get(lang, {}):
                    words[j] = self.rng.choice(self.cs[lang][core])
                    tgt = None  # morphology of a switched word is not observable
                targets.append(tgt)
            enc = self.tok(words, is_split_into_words=True, truncation=True, max_length=MAXLEN)
            ids = torch.tensor(enc.input_ids)
            word_ids = enc.word_ids()
            tag_pos, tag_tgt = [], []
            seen = set()
            for k, wi in enumerate(word_ids):
                if wi is None or wi in seen:
                    continue
                seen.add(wi)
                if targets[wi]:
                    v = torch.zeros(self.n_tags)
                    for t, val in targets[wi]:
                        v[t] = val
                    tag_pos.append(k)
                    tag_tgt.append(v)
            tags = torch.stack(tag_tgt) if tag_tgt else torch.zeros(0, self.n_tags)
            if not self.a.align:
                yield ids, tag_pos, tags
                continue
            g_ids, pairs = torch.zeros(0, dtype=torch.long), []
            if glossed and any(glossed):
                genc = self.tok([g or w for w, g in zip(words, glossed)], is_split_into_words=True, truncation=True,
                                max_length=MAXLEN)
                g_ids = torch.tensor(genc.input_ids)
                first_u, first_g = {}, {}
                for k, wi in enumerate(word_ids):
                    if wi is not None and wi not in first_u:
                        first_u[wi] = k
                for k, wi in enumerate(genc.word_ids()):
                    if wi is not None and wi not in first_g:
                        first_g[wi] = k
                pairs = [(first_u[wi], first_g[wi]) for wi, g in enumerate(glossed) if g and wi in first_u and wi in first_g]
            yield ids, tag_pos, tags, g_ids, pairs


def collate(batch, tok):
    mlm_prob = phase()[1]
    L_ = max(len(b[0]) for b in batch)
    ids = torch.full((len(batch), L_), tok.pad_token_id)
    for i, b in enumerate(batch):
        ids[i, :len(b[0])] = b[0]
    att = (ids != tok.pad_token_id).long()
    special = torch.isin(ids, torch.tensor(tok.all_special_ids))
    mask = (torch.rand(ids.shape) < mlm_prob) & ~special & att.bool()
    labels = ids.clone(); labels[~mask] = -100
    inp = ids.clone()
    r = torch.rand(ids.shape)
    inp[mask & (r < 0.8)] = tok.mask_token_id
    rnd = mask & (r >= 0.8) & (r < 0.9)
    inp[rnd] = torch.randint(len(tok), (int(rnd.sum()),))
    tb = torch.cat([torch.full((len(p),), i) for i, p in enumerate(b[1] for b in batch)]) if any(b[1] for b in batch) else torch.zeros(0, dtype=torch.long)
    tp = torch.tensor([k for b in batch for k in b[1]], dtype=torch.long)
    tt = torch.cat([b[2] for b in batch]) if len(tp) else torch.zeros(0, batch[0][2].shape[1])
    if len(batch[0]) == 3:
        return inp, att, labels, tb.long(), tp, tt
    # alignment: the English glosses (unmasked) and the aligned (sample, Uralic position, gloss position) triples
    have = [i for i, b in enumerate(batch) if b[4]]
    if not have:
        return inp, att, labels, tb.long(), tp, tt, None
    Lg = max(len(batch[i][3]) for i in have)
    g = torch.full((len(have), Lg), tok.pad_token_id)
    for r, i in enumerate(have):
        g[r, :len(batch[i][3])] = batch[i][3]
    trip = torch.tensor([(r, i, ku, kg) for r, i in enumerate(have) for ku, kg in batch[i][4]], dtype=torch.long)
    return inp, att, labels, tb.long(), tp, tt, (g, (g != tok.pad_token_id).long(), torch.tensor(have), trip)


class Model(L.LightningModule):
    def __init__(self, n_tags, lr, steps, fst_weight):
        super().__init__()
        self.save_hyperparameters()
        cfg = AutoConfig.from_pretrained(BASE)
        cfg.sparse_prediction = True
        self.mlm = AutoModelForMaskedLM.from_pretrained(BASE, config=cfg, attn_implementation="sdpa")
        self.tag_head = nn.Sequential(nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.GELU(), nn.Linear(cfg.hidden_size, n_tags))

    def on_train_batch_start(self, batch, batch_idx):
        STEP.value = self.global_step

    def training_step(self, batch, _):
        inp, att, labels, tb, tp, tt = batch[:6]
        al = batch[6] if len(batch) > 6 else None
        out = self.mlm(input_ids=inp, attention_mask=att, labels=labels, output_hidden_states=len(tp) > 0 or al is not None)
        loss = out.loss
        self.log("mlm", out.loss, prog_bar=True)
        self.log("mask_rate", phase()[1])
        if len(tp):
            h = out.hidden_states[-1][tb, tp]
            tag_loss = F.binary_cross_entropy_with_logits(self.tag_head(h).float(), tt)
            loss = loss + self.hparams.fst_weight * tag_loss
            self.log("tag", tag_loss, prog_bar=True)
        if al is not None:
            g, gatt, have, trip = al
            with torch.no_grad():  # the English gloss is the anchor
                hg = self.mlm.model(input_ids=g, attention_mask=gatt).last_hidden_state.detach()
            hu = out.hidden_states[-1]
            word = 1 - F.cosine_similarity(hu[trip[:, 1], trip[:, 2]].float(), hg[trip[:, 0], trip[:, 3]].float(), dim=-1).mean()
            mu = (hu[have] * att[have, :, None]).sum(1) / att[have].sum(1, keepdim=True)
            mg = (hg * gatt[..., None]).sum(1) / gatt.sum(1, keepdim=True)
            sent = 1 - F.cosine_similarity(mu.float(), mg.float(), dim=-1).mean()
            loss = loss + word + sent
            self.log("align_word", word, prog_bar=True)
            self.log("align_sent", sent)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01, betas=(0.9, 0.98))
        warm = int(0.06 * self.hparams.steps)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.0, 1 - s / self.hparams.steps) if s >= warm else (s + 1) / warm)
        return [opt], [{"scheduler": sched, "interval": "step"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fst", action="store_true")
    ap.add_argument("--cs", action="store_true")
    ap.add_argument("--par", action="store_true")
    ap.add_argument("--gloss", action="store_true")
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--keep_last", action="store_true", help="keep only the last checkpoint (saves disk space)")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--fst_weight", type=float, default=1.0)
    a = ap.parse_args()
    L.seed_everything(0)
    torch.set_float32_matmul_precision("high")
    tok = AutoTokenizer.from_pretrained(BASE)
    tags = json.load(open(f"{P}/tags.json"))
    TOTAL.value = a.steps
    dl = DataLoader(Stream(a, tok, len(tags)), batch_size=a.batch, num_workers=4,
                    collate_fn=lambda b: collate(b, tok), prefetch_factor=4, persistent_workers=True)
    model = Model(len(tags), a.lr, a.steps, a.fst_weight)
    out = f"{W}/models/{a.name}"
    ckpt = L.pytorch.callbacks.ModelCheckpoint(dirpath=f"{out}/ckpt", every_n_train_steps=1000, save_last=True,
                                         save_top_k=1 if a.keep_last else -1)
    trainer = L.Trainer(max_steps=a.steps, precision="bf16-mixed", accumulate_grad_batches=a.accum, gradient_clip_val=1.0, log_every_n_steps=50,
                        default_root_dir=out, callbacks=[ckpt], logger=L.pytorch.loggers.CSVLogger(out))
    last = f"{out}/ckpt/last.ckpt"
    trainer.fit(model, dl, ckpt_path=last if os.path.exists(last) else None)
    model.mlm.save_pretrained(f"{out}/hf")
    tok.save_pretrained(f"{out}/hf")
    torch.save(model.tag_head.state_dict(), f"{out}/tag_head.pt")
    json.dump(vars(a), open(f"{out}/args.json", "w"))


if __name__ == "__main__":
    main()
