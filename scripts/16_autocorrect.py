"""Keyboard autocorrection with FST-validated candidates and left-context encoder scoring.

Setting: the user has typed a sentence up to the current word; the system sees the left context and the typed
word and outputs a correction (or keeps the word). Noise:
  diacritics  letters missing from the keyboard layout are typed as their base letter (e.g. ӧ->о, õ->o, ʹ dropped)
  typo        one adjacent-key substitution / deletion / insertion / transposition
  clean       the word is typed correctly (measures over-correction)
Candidates: the typed word, its letter restorations and its edit-1 neighbours, kept if the corpus lexicon contains
them or the FST (pyhfst) accepts them. Ranking: a*encoder + b*log-frequency + c*edit cost, tuned on UD *train*
sentences; test text = UD test sentences (removed from all pretraining data).

Stages:  candidates  (CPU, pyhfst, cached)  ->  score <encoder> <name>  (GPU)"""
import argparse, collections, functools, glob, itertools, json, math, os, random, re, sys, unicodedata
from multiprocessing import Pool
import numpy as np, pandas as pd, torch

W = os.environ.get("LEXIBERT_WORK", "work")
OUT = os.environ.get("AC_OUT", f"{W}/data/autocorrect")
UD = {"myv": ["Erzya-JR"], "mdf": ["Moksha-JR"], "kpv": ["Komi_Zyrian-Lattice", "Komi_Zyrian-IKDP"], "koi": ["Komi_Permyak-UH"],
      "sms": ["Skolt_Sami-Giellagas"], "olo": ["Livvi-KKPP"], "krl": ["Karelian-KKPP"], "sme": ["North_Sami-Giella"],
      "vep": ["Veps-VWT"], "yrk": ["Nenets-Tundra"]}
CYRILLIC = {"myv", "mdf", "kpv", "koi", "yrk"}
RU_KEYS = ["йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю"]
FI_KEYS = ["qwertyuiopå", "asdfghjklöä", "zxcvbnm"]
MAX_TARGETS, MAX_CANDS, SEED = 1500, 24, 0


def layout(lang):
    rows = RU_KEYS if lang in CYRILLIC else FI_KEYS
    keys = set("".join(rows))
    adj = collections.defaultdict(set)
    for r, row in enumerate(rows):
        for i, ch in enumerate(row):
            for rr in (r - 1, r, r + 1):
                if 0 <= rr < len(rows):
                    for j in (i - 1, i, i + 1):
                        if 0 <= j < len(rows[rr]) and rows[rr][j] != ch:
                            adj[ch].add(rows[rr][j])
    return keys, adj


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


def typo(word, adj, rng):
    i = rng.randrange(len(word))
    ch = word[i].lower()
    kind = rng.choices(["sub", "del", "ins", "swap"], [0.6, 0.15, 0.15, 0.1])[0]
    if kind == "sub" and adj.get(ch):
        return word[:i] + rng.choice(sorted(adj[ch])) + word[i + 1:]
    if kind == "del" and len(word) > 3:
        return word[:i] + word[i + 1:]
    if kind == "ins" and adj.get(ch):
        return word[:i] + rng.choice(sorted(adj[ch])) + word[i:]
    if kind == "swap" and i < len(word) - 1 and word[i] != word[i + 1]:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + rng.choice(sorted(adj[ch])) + word[i + 1:] if adj.get(ch) else word[:i] + word[i + 1:]


def read_ud(lang, split):
    sents = []
    for tb in UD[lang]:
        for f in glob.glob(f"{W}/data/ud/UD_{tb}/*-ud-{split}.conllu"):
            cur = []
            for line in open(f, encoding="utf-8"):
                p = line.rstrip("\n").split("\t")
                if len(p) == 10 and p[0].isdigit():
                    cur.append((p[1], p[3]))
                elif not line.strip() and cur:
                    sents.append(cur); cur = []
            if cur:
                sents.append(cur)
    return sents


# ---------------------------------------------------------------- candidates (CPU, pyhfst)
_G = {}


def _init(lang):
    from uralicNLP import uralicApi
    fst = pd.read_parquet(f"{W}/data/fst/{lang}.parquet", columns=["word", "analyses", "freq"])
    freq = dict(zip(fst.word, fst.freq))
    script = "CYRILLIC" if lang in CYRILLIC else "LATIN"
    chars = collections.Counter()
    for w, n in freq.items():
        for ch in w.lower():
            chars[ch] += n
    total = sum(chars.values())
    alphabet = sorted(ch for ch, n in chars.items() if ch.isalpha() and n / total > 1e-4
                      and unicodedata.name(ch, "").startswith(script))
    trigrams = {f"<{w.lower()}>"[i:i + 3] for w, n in freq.items() if n > 0 for i in range(len(w))}
    _G.update(lang=lang, api=uralicApi, freq=freq, keys=layout(lang)[0], alphabet=alphabet, trigrams=trigrams, cache={})
    # restoration index: keyboard form -> lexicon words that produce it
    idx = collections.defaultdict(list)
    for w, n in freq.items():
        if n > 0:
            idx[keyboard_form(w, _G["keys"])].append(w)
    _G["restore"] = idx


def fst_ok(w):
    c = _G["cache"]
    if w not in c:
        c[w] = bool(_G["api"].analyze(w, _G["lang"])) or (w[:1].isupper() and bool(_G["api"].analyze(w.lower(), _G["lang"])))
    return c[w]


def restorations(typed):
    """Letter restorations: lexicon words with this keyboard form, plus FST-valid expansions (bounded)."""
    out = set(_G["restore"].get(typed, []))
    inv = collections.defaultdict(set)
    for special, base in SPECIAL_BASE.items():
        if base:
            inv[base].add(special)
    slots = [[ch] + sorted(inv.get(ch.lower(), set()) - _G["keys"]) for ch in typed]
    combos = 1
    for s in slots:
        combos *= len(s)
    if combos <= 256:
        for cand in map("".join, itertools.product(*slots)):
            if cand != typed and plausible(cand) and fst_ok(cand):
                out.add(cand)
    return out


def plausible(w):
    """Cheap pre-filter before an FST lookup: every character trigram must occur in the lexicon."""
    t, m = _G["trigrams"], f"<{w.lower()}>"
    return all(m[i:i + 3] in t for i in range(len(m) - 2))


def edits1(w):
    letters = _G["alphabet"]
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    out = {a + b[1:] for a, b in splits if b}
    out |= {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}
    out |= {a + c + b[1:] for a, b in splits if b for c in letters}
    out |= {a + c + b for a, b in splits for c in letters}
    out.discard(w)
    return out


def candidates(job):
    typed, cond = job
    freq = _G["freq"]
    cands = {typed: 0}
    for c in restorations(typed):
        cands.setdefault(c, 0)
    for c in edits1(typed):
        if c in cands:
            continue
        in_lex = freq.get(c, 0) > 0
        if in_lex or (plausible(c) and fst_ok(c)):
            cands[c] = 1
    # keep the typed word + the best candidates by (edit cost, frequency); record their provenance
    ranked = sorted(cands.items(), key=lambda kv: (kv[0] != typed, kv[1], -freq.get(kv[0], 0)))[:MAX_CANDS]
    return [(c, e, freq.get(c, 0), int(freq.get(c, 0) > 0), int(fst_ok(c))) for c, e in ranked]


def build(lang, split, max_targets, rng):
    keys, adj = layout(lang)
    rows = []
    for si, sent in enumerate(read_ud(lang, split)):
        for wi, (form, upos) in enumerate(sent):
            if len(form) < 3 or not form.isalpha() or upos in ("PROPN", "PUNCT", "X", "SYM", "NUM"):
                continue
            ctx = " ".join(f for f, _ in sent[:wi])
            kf = keyboard_form(form, keys)
            if kf != form:
                rows.append(dict(sent=si, pos=wi, gold=form, typed=kf, cond="diacritics", upos=upos, ctx=ctx))
            t = typo(form, adj, rng)
            if t != form:
                rows.append(dict(sent=si, pos=wi, gold=form, typed=t, cond="typo", upos=upos, ctx=ctx))
            rows.append(dict(sent=si, pos=wi, gold=form, typed=form, cond="clean", upos=upos, ctx=ctx))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    parts = [g.sample(min(len(g), max_targets), random_state=SEED) for _, g in df.groupby("cond")]
    return pd.concat(parts, ignore_index=True) if parts else df


def stage_candidates(langs):
    os.makedirs(OUT, exist_ok=True)
    for lang in langs:
        rng = random.Random(SEED)
        for split, n in (("test", MAX_TARGETS), ("train", 300)):
            path = f"{OUT}/{lang}.{split}.parquet"
            if os.path.exists(path):
                continue
            df = build(lang, split, n, rng)
            if not len(df):
                continue
            with Pool(int(os.environ.get("AC_PROCS", 12)), initializer=_init, initargs=(lang,)) as pool:
                cands = pool.map(candidates, list(zip(df.typed, df.cond)), chunksize=8)
            df["cands"] = [json.dumps(c, ensure_ascii=False) for c in cands]
            freq = dict(pd.read_parquet(f"{W}/data/fst/{lang}.parquet", columns=["word", "freq"]).values)
            df["gold_freq"] = df.gold.map(lambda w: freq.get(w, 0))
            df.to_parquet(path, index=False)
            rec = np.mean([any(c[0] == g for c in json.loads(cs)) for g, cs in zip(df.gold, df.cands)])
            print(lang, split, len(df), df.cond.value_counts().to_dict(), "candidate recall", round(100 * rec, 1), flush=True)


# ---------------------------------------------------------------- scoring (GPU)
@torch.no_grad()
def lm_scores(df, encoder, bs=192, return_counts=False):
    """Pseudo-log-likelihood of each candidate given only the left context (sub-tokens masked one at a time)."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(encoder)
    model = AutoModelForMaskedLM.from_pretrained(encoder, dtype=torch.bfloat16).cuda().eval()
    jobs, owners = [], []
    for r, (ctx, cs) in enumerate(zip(df.ctx, df.cands)):
        left = tok(ctx, add_special_tokens=False).input_ids[-60:]
        for k, c in enumerate(json.loads(cs)):
            ids = tok(" " + c[0] if ctx else c[0], add_special_tokens=False).input_ids
            seq = [tok.cls_token_id] + left + ids + [tok.sep_token_id]
            start = 1 + len(left)
            for j in range(len(ids)):
                jobs.append((seq, start + j, ids[j])); owners.append((r, k))
    scores = collections.defaultdict(float)
    counts = collections.Counter(owners)
    for i in range(0, len(jobs), bs):
        chunk = jobs[i:i + bs]
        L = max(len(s) for s, _, _ in chunk)
        x = torch.full((len(chunk), L), tok.pad_token_id)
        att = torch.zeros(len(chunk), L, dtype=torch.long)
        for b, (s, p, _) in enumerate(chunk):
            x[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1; x[b, p] = tok.mask_token_id
        hidden = model.model(input_ids=x.cuda(), attention_mask=att.cuda())[0]
        pos = torch.tensor([p for _, p, _ in chunk], device=hidden.device)
        h = hidden[torch.arange(len(chunk), device=hidden.device), pos]  # only the masked positions
        logp = model.decoder(model.head(h)).float().log_softmax(-1)
        tgt = torch.tensor([t for _, _, t in chunk], device=hidden.device)
        vals = logp[torch.arange(len(chunk), device=hidden.device), tgt].tolist()
        for b in range(len(chunk)):
            scores[owners[i + b]] += vals[b]
    return (scores, counts) if return_counts else scores


def choose(df, lm, a, b, c, use_fst=True, norm=False):
    """lm = (sums, counts); norm=True scores a candidate by its mean log-probability per sub-token."""
    sums, counts = lm if isinstance(lm, tuple) else (lm, {})
    out = []
    for r, cs in enumerate(df.cands):
        best, best_s = None, -1e18
        for k, (cand, edit, freq, in_lex, fst) in enumerate(json.loads(cs)):
            if k > 0 and not in_lex and not (use_fst and fst):
                continue
            v = sums.get((r, k), 0.0)
            if norm:
                v /= max(1, counts.get((r, k), 1))
            s = a * v + b * math.log1p(freq) - c * edit
            if s > best_s:
                best, best_s = cand, s
        out.append(best)
    return out


def metrics(df, pred):
    df = df.assign(pred=pred, hit=lambda d: d.pred == d.gold)
    res = {cond: round(100 * g.hit.mean(), 1) for cond, g in df.groupby("cond")}
    bands = pd.cut(df.gold_freq, [-1, 0, 10, 1e12], labels=["unseen", "rare(1-10)", "frequent"])
    res["by_freq"] = {f"{cond}/{band}": round(100 * g.hit.mean(), 1) for (cond, band), g in df.groupby(["cond", bands], observed=True) if len(g) >= 20}
    res["by_upos"] = {f"{cond}/{u}": round(100 * g.hit.mean(), 1) for (cond, u), g in df.groupby(["cond", "upos"]) if len(g) >= 30}
    return res


def stage_score(langs, encoder, name):
    os.makedirs(f"{W}/runs/autocorrect", exist_ok=True)
    dev = {l: pd.read_parquet(f"{OUT}/{l}.train.parquet") for l in langs if os.path.exists(f"{OUT}/{l}.train.parquet")}
    dev_lm = {l: lm_scores(d, encoder, return_counts=True) for l, d in dev.items()}
    grid = [(a, b, c, norm) for a in (0.0, 0.1, 0.25, 0.5, 1.0) for b in (0.0, 0.25, 0.5, 1.0, 2.0)
            for c in (0.0, 1.0, 2.0, 5.0, 10.0) for norm in (False, True) if a > 0 or not norm]

    def dev_acc(a, b, c, norm, use_fst):
        # mean over conditions so that keeping correct words and fixing errors count equally
        accs = []
        for l, d in dev.items():
            m = metrics(d, choose(d, dev_lm[l], a, b, c, use_fst, norm))
            accs += [m[k] for k in ("diacritics", "typo", "clean") if k in m]
        return np.mean(accs)

    # frequency only (lexicon candidates) | encoder with lexicon candidates | encoder with lexicon + FST candidates
    systems = {}
    for label, use_lm, use_fst in [("freq_lexicon", False, False), ("lm_lexicon", True, False), ("lm_fst", True, True)]:
        options = [g for g in grid if (g[0] > 0) == use_lm]
        systems[label] = (max(options, key=lambda g: dev_acc(*g, use_fst)), use_fst)
        print(name, label, "weights (a, b, c, norm)", systems[label][0], flush=True)
    res = {"weights": {k: list(v[0]) for k, v in systems.items()}}
    for l in langs:
        p = f"{OUT}/{l}.test.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        lm = lm_scores(d, encoder, return_counts=True)
        res[l] = {label: metrics(d, choose(d, lm, *w[:3], use_fst, w[3])) for label, (w, use_fst) in systems.items()}
        res[l]["keep_typed"] = metrics(d, list(d.typed))
        res[l]["candidate_recall"] = round(100 * np.mean([any(c[0] == g for c in json.loads(cs)) for g, cs in zip(d.gold, d.cands)]), 1)
        print(name, l, {k: {c: v[c] for c in ("diacritics", "typo", "clean") if c in v} for k, v in res[l].items() if isinstance(v, dict)}, flush=True)
    json.dump(res, open(f"{W}/runs/autocorrect/{name}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["candidates", "score"])
    ap.add_argument("--encoder")
    ap.add_argument("--name")
    ap.add_argument("--langs", nargs="+", default=list(UD))
    a = ap.parse_args()
    stage_candidates(a.langs) if a.stage == "candidates" else stage_score(a.langs, a.encoder, a.name)
