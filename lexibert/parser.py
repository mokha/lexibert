"""UD parser: LexiBERT + UPOS / UFeats / lemma classifiers and a biaffine dependency parser (gold tokenization).
The lemma can optionally be restricted to the lemmas given by the FST (Lem_FST in the paper)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from .common import autocast, default_device, encoder_from_config, fst_analyses, load, load_weights


def lemma_rule(form, lemma):
    f, l = form.lower(), lemma.lower()
    p = 0
    while p < min(len(f), len(l)) and f[p] == l[p]:
        p += 1
    return f"{'U' if lemma[:1].isupper() else 'L'}|{len(f) - p}|{l[p:]}"


def apply_rule(form, rule):
    case, k, s = rule.split("|", 2)
    k = int(k)
    if k > len(form):
        return form
    out = (form.lower()[:-k] if k else form.lower()) + s
    return out[:1].upper() + out[1:] if case == "U" else out


def mst_heads(arc, k):
    """The highest scoring tree with exactly one root (Chu-Liu-Edmonds, as in Stanza)."""
    from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root
    sc = arc[:k, :k + 1].float().cpu().clone()
    sc[torch.arange(k), torch.arange(1, k + 1)] = -float("inf")
    scores = np.zeros((k + 1, k + 1))
    scores[1:] = F.log_softmax(sc, -1).numpy()
    return chuliu_edmonds_one_root(scores)[1:].tolist()


class Parser(nn.Module):
    def __init__(self, folder, vocabs):
        super().__init__()
        self.enc = encoder_from_config(folder)
        H = self.enc.config.hidden_size
        self.vocabs = vocabs
        self.lemma_ids = {r: i for i, r in enumerate(vocabs["lemma"])}
        self.mix = nn.Parameter(torch.zeros(self.enc.config.num_hidden_layers + 1))
        self.drop = nn.Dropout(0.2)
        self.upos = nn.Linear(H, len(vocabs["upos"]))
        self.feats = nn.Linear(H, len(vocabs["feats"]))
        self.lemma = nn.Linear(H, len(vocabs["lemma"]))
        self.arc_h, self.arc_d = nn.Linear(H, 384), nn.Linear(H, 384)
        self.rel_h, self.rel_d = nn.Linear(H, 128), nn.Linear(H, 128)
        self.arc_W = nn.Parameter(torch.zeros(384, 384)); self.arc_b = nn.Linear(384, 1, bias=False)
        self.rel_W = nn.Parameter(torch.zeros(len(vocabs["rel"]), 129, 129))
        self.root = nn.Parameter(torch.zeros(H))

    @classmethod
    def from_pretrained(cls, path, device=None):
        folder, meta, vocabs, tok = load(path)
        device = default_device(device)
        model = load_weights(cls(folder, vocabs), folder, device)
        model.tok, model.meta, model.device = tok, meta, device
        model._fst = {}
        return model

    def forward(self, ids, att, first, wmask):
        out = self.enc(input_ids=ids, attention_mask=att, output_hidden_states=True)
        h = (F.softmax(self.mix, 0)[:, None, None, None] * torch.stack(out.hidden_states, 0)).sum(0)
        B = first.shape[0]
        w = torch.cat([self.root.expand(B, 1, -1), h[torch.arange(B)[:, None], first]], 1)
        tok = w[:, 1:]
        ah, ad = F.gelu(self.arc_h(w)), F.gelu(self.arc_d(w))
        arc = torch.einsum("bie,ef,bjf->bij", ad, self.arc_W, ah) + self.arc_b(ah).transpose(1, 2)
        full = torch.cat([torch.ones_like(wmask[:, :1]), wmask], 1)
        arc = arc.masked_fill(~full[:, None, :], -1e4)
        rh = torch.cat([F.gelu(self.rel_h(w)), torch.ones_like(w[..., :1])], -1)
        rd = torch.cat([F.gelu(self.rel_d(w)), torch.ones_like(w[..., :1])], -1)
        return self.upos(tok), self.feats(tok), self.lemma(tok), arc[:, 1:], rh, rd[:, 1:]

    def fst_lemmas(self, lang, form):
        if lang not in self._fst:
            self._fst[lang] = fst_analyses(lang)
        return {a.split("+")[0] for a in self._fst[lang](form) if "#" not in a}

    def fst_lemma(self, form, pred, cands, logits):
        """The FST lemma whose edit rule the lemma classifier scores highest, otherwise the shortest one."""
        if not cands or pred in cands:
            return pred
        scored = [(logits[self.lemma_ids[r]].item(), c) for c in cands
                  if (r := lemma_rule(form, c)) in self.lemma_ids and self.lemma_ids[r]]
        return max(scored)[1] if scored else min(cands, key=lambda c: (len(c), c))

    @torch.no_grad()
    def parse(self, sentences, fst_lang=None, batch_size=32, max_length=256):
        """Parse tokenized sentences (lists of word forms). With `fst_lang` (e.g. 'myv'), the lemma is restricted to
        the lemmas of the UralicNLP FST of that language. Returns a list of sentences, each a list of dicts."""
        v, res = self.vocabs, []
        for i in range(0, len(sentences), batch_size):
            chunk = [list(s) for s in sentences[i:i + batch_size]]
            encs = [self.tok(s, is_split_into_words=True, truncation=True, max_length=max_length) for s in chunk]
            firsts = []
            for s, e in zip(chunk, encs):
                f = [-1] * len(s)
                for k, wi in enumerate(e.word_ids()):
                    if wi is not None and f[wi] < 0:
                        f[wi] = k
                firsts.append(f)
            B, Lt, Lw = len(chunk), max(len(e.input_ids) for e in encs), max(len(s) for s in chunk)
            ids = torch.full((B, Lt), self.tok.pad_token_id); att = torch.zeros(B, Lt, dtype=torch.long)
            first = torch.zeros(B, Lw, dtype=torch.long); wmask = torch.zeros(B, Lw, dtype=torch.bool)
            for b, (e, f) in enumerate(zip(encs, firsts)):
                ids[b, :len(e.input_ids)] = torch.tensor(e.input_ids); att[b, :len(e.input_ids)] = 1
                ft = torch.tensor(f); first[b, :len(f)] = ft.clamp(min=0); wmask[b, :len(f)] = ft >= 0
            ids, att, first, wmask = (x.to(self.device) for x in (ids, att, first, wmask))
            with autocast(self.device):
                pu, pf, pl, arc, rh, rd = self(ids, att, first, wmask)
                heads = torch.zeros(B, Lw, dtype=torch.long)
                for b in range(B):
                    k = int(wmask[b].sum())  # truncation drops words only at the end
                    if k:
                        heads[b, :k] = torch.tensor(mst_heads(arc[b], k))
                heads = heads.to(self.device)
                hsel = rh[torch.arange(B)[:, None], heads]
                rels = torch.einsum("bie,ref,bif->bir", rd, self.rel_W, hsel).argmax(-1).cpu()
            pu, pf, pl, pl_logits = pu.argmax(-1).cpu(), pf.argmax(-1).cpu(), pl.argmax(-1).cpu(), pl.float().cpu()
            for b, s in enumerate(chunk):
                out = []
                for j, form in enumerate(s):
                    if firsts[b][j] < 0:  # beyond max_length
                        out.append(dict(id=j + 1, form=form, lemma="_", upos="_", feats="_", head=0, deprel="_"))
                        continue
                    lemma = apply_rule(form, v["lemma"][pl[b, j]]) if pl[b, j] else form
                    if fst_lang:
                        lemma = self.fst_lemma(form, lemma, self.fst_lemmas(fst_lang, form), pl_logits[b, j])
                    out.append(dict(id=j + 1, form=form, lemma=lemma, upos=v["upos"][pu[b, j]], feats=v["feats"][pf[b, j]],
                                    head=int(heads[b, j]), deprel=v["rel"][rels[b, j]]))
                res.append(out)
        return res


def to_conllu(parsed):
    lines = []
    for s in parsed:
        for t in s:
            lines.append("\t".join(map(str, [t["id"], t["form"], t["lemma"], t["upos"], "_", t["feats"], t["head"],
                                             t["deprel"], "_", "_"])))
        lines.append("")
    return "\n".join(lines) + "\n"


def read_input(f):
    """CoNLL-U (the FORM column is used; multiword token lines and empty nodes are skipped) or one tokenized
    sentence per line."""
    text = f.read()
    if any(len(l.split("\t")) == 10 for l in text.splitlines()[:50]):
        sents, cur = [], []
        for line in text.splitlines():
            p = line.split("\t")
            if not line.strip():
                if cur:
                    sents.append(cur)
                cur = []
            elif len(p) == 10 and p[0].isdigit():
                cur.append(p[1])
        if cur:
            sents.append(cur)
        return sents
    return [l.split() for l in text.splitlines() if l.strip()]
