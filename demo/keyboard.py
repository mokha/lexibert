"""Keyboard autocorrection as in the paper (scripts/16_autocorrect.py): the candidates of a typed word are the word
itself, its versions with the letters that are missing from a Finnish or Russian keyboard restored, and the words
within one edit, kept if they are in the corpus lexicon or accepted by the FST. They are ranked by
a * (log-probability of the candidate after the preceding words, from the encoder) + b * log(1 + frequency)
- c * (number of edits), with the weights tuned on the UD training sentences."""
import collections, gzip, itertools, json, math, os, unicodedata
import torch

CYRILLIC = {"myv", "mdf", "kpv", "koi", "udm", "mhr", "mrj", "yrk"}
RU_KEYS = ["йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю"]
FI_KEYS = ["qwertyuiopå", "asdfghjklöä", "zxcvbnm"]
SPECIAL_BASE = {"ӧ": "о", "ӓ": "а", "ӱ": "у", "ӹ": "ы", "і": "и", "ӥ": "и", "ӈ": "н", "ҥ": "н", "ӝ": "ж", "ӟ": "з",
                "ӵ": "ч", "ӭ": "э", "ӛ": "э", "ә": "э", "ʼ": "", "ʹ": "", "ꞌ": "", "’": "", "ǩ": "k", "ǧ": "g", "ǥ": "g",
                "ʒ": "z", "ǯ": "z", "đ": "d", "ŋ": "n", "õ": "o", "â": "a", "č": "c", "š": "s", "ž": "z", "ȯ": "o",
                "ȱ": "o", "ǟ": "a", "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u", "ü": "y", "ʾ": ""}


def keyboard_form(word, keys):
    """How the word comes out on a keyboard without the language's special letters."""
    out = []
    for ch in word:
        low = ch.lower()
        if low in keys or not low.isalpha():
            out.append(ch)
            continue
        base = SPECIAL_BASE.get(low)
        if base is None:
            base = unicodedata.normalize("NFD", low)[0]
        out.append(base.upper() if ch.isupper() and base else base)
    return "".join(out)


class Keyboard:
    def __init__(self, folder, lang, fst_accepts):
        cfg = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))
        self.a, self.b, self.c, self.max_cands = cfg["a"], cfg["b"], cfg["c"], cfg["max_candidates"]
        self.lang, self.fst_accepts = lang, fst_accepts
        self.freq = {}
        with gzip.open(os.path.join(folder, "lexicons", f"{lang}.tsv.gz"), "rt", encoding="utf-8") as f:
            for line in f:
                w, n = line.rstrip("\n").split("\t")
                self.freq[w] = int(n)
        self.keys = set("".join(RU_KEYS if lang in CYRILLIC else FI_KEYS))
        script = "CYRILLIC" if lang in CYRILLIC else "LATIN"
        chars = collections.Counter()
        for w, n in self.freq.items():
            for ch in w.lower():
                chars[ch] += n
        total = sum(chars.values())
        self.alphabet = sorted(ch for ch, n in chars.items() if ch.isalpha() and n / total > 1e-4
                               and unicodedata.name(ch, "").startswith(script))
        self.trigrams = {f"<{w.lower()}>"[i:i + 3] for w in self.freq for i in range(len(w))}
        self.restore = collections.defaultdict(list)
        for w in self.freq:
            self.restore[keyboard_form(w, self.keys)].append(w)
        self.cache = {}

    def fst_ok(self, w):
        if w not in self.cache:
            self.cache[w] = bool(self.fst_accepts(w)) or (w[:1].isupper() and bool(self.fst_accepts(w.lower())))
        return self.cache[w]

    def plausible(self, w):
        m = f"<{w.lower()}>"
        return all(m[i:i + 3] in self.trigrams for i in range(len(m) - 2))

    def restorations(self, typed):
        out = set(self.restore.get(typed, []))
        inv = collections.defaultdict(set)
        for special, base in SPECIAL_BASE.items():
            if base:
                inv[base].add(special)
        slots = [[ch] + sorted(inv.get(ch.lower(), set()) - self.keys) for ch in typed]
        if math.prod(len(s) for s in slots) <= 256:
            for cand in map("".join, itertools.product(*slots)):
                if cand != typed and self.plausible(cand) and self.fst_ok(cand):
                    out.add(cand)
        return out

    def edits1(self, w):
        splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        out = {a + b[1:] for a, b in splits if b}
        out |= {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}
        out |= {a + c + b[1:] for a, b in splits if b for c in self.alphabet}
        out |= {a + c + b for a, b in splits for c in self.alphabet}
        out.discard(w)
        return out

    def candidates(self, typed):
        """(candidate, edits, frequency) for the typed word, at most max_candidates, the typed word first."""
        cands = {typed: 0}
        for c in self.restorations(typed):
            cands.setdefault(c, 0)
        for c in self.edits1(typed):
            if c not in cands and (self.freq.get(c, 0) > 0 or (self.plausible(c) and self.fst_ok(c))):
                cands[c] = 1
        ranked = sorted(cands.items(), key=lambda kv: (kv[0] != typed, kv[1], -self.freq.get(kv[0], 0)))[:self.max_cands]
        # the typed word itself is always a candidate; other words must be in the lexicon or accepted by the FST
        return [(c, e, self.freq.get(c, 0)) for k, (c, e) in enumerate(ranked)
                if k == 0 or self.freq.get(c, 0) > 0 or self.fst_ok(c)]

    @torch.no_grad()
    def lm_scores(self, tok, model, context, cands, batch_size=64):
        """Log-probability of each candidate after the left context, its sub-words masked one at a time."""
        left = tok(context, add_special_tokens=False).input_ids[-60:] if context else []
        jobs = []
        for k, cand in enumerate(cands):
            ids = tok(" " + cand if context else cand, add_special_tokens=False).input_ids
            seq = [tok.cls_token_id] + left + ids + [tok.sep_token_id]
            for j in range(len(ids)):
                jobs.append((seq, 1 + len(left) + j, ids[j], k))
        scores = [0.0] * len(cands)
        for i in range(0, len(jobs), batch_size):
            chunk = jobs[i:i + batch_size]
            L = max(len(s) for s, _, _, _ in chunk)
            x = torch.full((len(chunk), L), tok.pad_token_id)
            att = torch.zeros(len(chunk), L, dtype=torch.long)
            for r, (s, p, _, _) in enumerate(chunk):
                x[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1; x[r, p] = tok.mask_token_id
            logits = model(input_ids=x, attention_mask=att).logits
            logp = logits[torch.arange(len(chunk)), torch.tensor([p for _, p, _, _ in chunk])].float().log_softmax(-1)
            for r, (_, _, t, k) in enumerate(chunk):
                scores[k] += float(logp[r, t])
        return scores

    def correct(self, tok, model, context, typed):
        """The best candidate and all candidates with their scores, best first."""
        cands = self.candidates(typed)
        lm = self.lm_scores(tok, model, context, [c for c, _, _ in cands])
        scored = [(c, self.a * s + self.b * math.log1p(f) - self.c * e, s, f, e) for (c, e, f), s in zip(cands, lm)]
        scored.sort(key=lambda x: -x[1])
        return scored[0][0], scored
