"""Build the unknown-word analysis dataset (form -> lemma edit rule, POS, tags, contlex).

Examples come from corpus types the FST *can* analyse; contlex is joined from Ve'rdd.
Splits are grouped by lemma so test lemmas (and all their forms) are unseen in training.
Also writes the real-OOV test set: UD tokens the FST cannot analyse (gold lemma/UPOS)."""
import glob, hashlib, json, os, re, sys
from collections import Counter
import pandas as pd

W = os.environ.get("LEXIBERT_WORK", "work")
OUT = f"{W}/data/oov"
UD = {"myv": "Erzya-JR", "mdf": "Moksha-JR", "kpv": "Komi_Zyrian-*", "koi": "Komi_Permyak-UH",
      "sms": "Skolt_Sami-Giellagas", "olo": "Livvi-KKPP", "krl": "Karelian-KKPP", "sme": "North_Sami-Giella",
      "vep": "Veps-VWT", "yrk": "Nenets-Tundra"}
POS_TAGS = {"N", "V", "A", "Adv", "Pron", "Num", "Adp", "Po", "Pr", "CC", "CS", "Pcle", "Interj", "Det", "Prop"}
MAX_TYPES = 120_000


def _h(x):
    return int(hashlib.md5(x.encode()).hexdigest(), 16)


SPLIT = {}


def split_of(lemma):
    if lemma in SPLIT:
        return SPLIT[lemma]
    h = _h(lemma) % 10
    return "test" if h == 0 else "dev" if h == 1 else "train"


def stratify(lemma_strata):
    """Assign lemmas to splits within each (POS, contlex) stratum: 10% test, 10% dev, >=1 test if stratum >= 3."""
    SPLIT.clear()
    groups = {}
    for lemma, stratum in lemma_strata.items():
        groups.setdefault(stratum, []).append(lemma)
    for lemmas in groups.values():
        lemmas.sort(key=_h)
        n = len(lemmas)
        n_test = max(1, round(0.1 * n)) if n >= 3 else 0
        n_dev = max(1, round(0.1 * n)) if n >= 5 else 0
        for i, l in enumerate(lemmas):
            SPLIT[l] = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"


def parse(analysis):
    """'кудо+N+Sg+Nom' -> (lemma, pos, tags). None for compounds/derivations/errors."""
    if "#" in analysis or "+Der" in analysis or "+Err" in analysis or "+Use/" in analysis:
        return None
    parts = analysis.split("+")
    lemma, tags = parts[0], [t for t in parts[1:] if t]
    if not lemma or not tags:
        return None
    pos_i = next((i for i, t in enumerate(tags) if t in POS_TAGS), None)
    if pos_i is None:
        return None
    pos = tags[pos_i]
    if pos == "N" and "Prop" in tags:
        pos = "Prop"
    rest = [t for i, t in enumerate(tags) if i != pos_i and t != "Prop"]
    return lemma, pos, "+".join(rest)


def edit_rule(form, lemma):
    """Casing flag + suffix rule: strip k chars from the lowered form, append s."""
    case = "U" if form[:1].isupper() and lemma[:1].isupper() else "L" if form[:1].isupper() else "K"
    f = form.lower() if case == "L" else form
    p = 0
    while p < min(len(f), len(lemma)) and f[p] == lemma[p]:
        p += 1
    return f"{case}|{len(f) - p}|{lemma[p:]}"


def apply_rule(form, rule):
    case, k, s = rule.split("|", 2)
    f = form.lower() if case == "L" else form
    k = int(k)
    return (f[:-k] if k else f) + s if k <= len(f) else None


def contlex_map(lang):
    lx = pd.read_parquet(f"{W}/data/verdd/lexemes.parquet", columns=["lang", "lexeme", "pos", "contlex"])
    lx = lx[(lx.lang == lang) & lx.contlex.fillna("").str.match(r".*[A-Za-zА-Яа-я]{2}")]
    m = {}
    for lexeme, pos, c in zip(lx.lexeme, lx.pos, lx.contlex):
        m.setdefault((lexeme, pos or ""), c)
        m.setdefault((lexeme, ""), c)
    return m


def build(lang):
    fst = pd.read_parquet(f"{W}/data/fst/{lang}.parquet")
    fst = fst[fst.analyses.map(len) > 0].sort_values("freq", ascending=False)
    cmap = contlex_map(lang)
    strata = {}
    for analyses in fst.analyses.head(4 * MAX_TYPES):
        for p in filter(None, map(parse, analyses)):
            strata.setdefault(p[0], (p[1], cmap.get((p[0], p[1]), cmap.get((p[0], ""), ""))))
    stratify(strata)
    rows, n_forms = [], 0
    for word, analyses, freq in zip(fst.word, fst.analyses, fst.freq):
        if n_forms >= MAX_TYPES:
            break
        parsed = {p for p in map(parse, analyses) if p}
        if not parsed:
            continue
        splits = {split_of(l) for l, _, _ in parsed}
        if len(splits) > 1:  # form shared by lemmas in different splits: drop to avoid leakage
            continue
        gold = sorted(parsed)
        n_forms += 1
        for lemma, pos, tags in gold:
            rows.append(dict(lang=lang, form=word, freq=freq, lemma=lemma, pos=pos, tags=tags,
                             rule=edit_rule(word, lemma), split=splits.pop() if len(splits) == 1 else split_of(lemma),
                             contlex=cmap.get((lemma, pos), cmap.get((lemma, ""), "")),
                             stratum_size=0,
                             gold=json.dumps(gold, ensure_ascii=False)))
            splits = {split_of(lemma)}
    df = pd.DataFrame(rows)
    assert all(apply_rule(f, r) == l for f, r, l in zip(df.form.head(2000), df.rule.head(2000), df.lemma.head(2000)))
    return df


def ud_oov(lang, split="test"):
    if lang not in UD:
        return None
    fst = pd.read_parquet(f"{W}/data/fst/{lang}.parquet")
    unknown = set(fst.word[fst.analyses.map(len) == 0])
    rows = []
    for f in glob.glob(f"{W}/data/ud/UD_{UD[lang]}/*-ud-{split}.conllu"):
        sent = []
        for line in open(f, encoding="utf-8"):
            if line.startswith("# text ="):
                sent = line.split("=", 1)[1].strip()
            p = line.rstrip("\n").split("\t")
            if len(p) == 10 and p[0].isdigit() and p[1] in unknown and p[3] not in ("PUNCT", "NUM", "SYM", "X"):
                rows.append(dict(lang=lang, form=p[1], lemma=p[2], upos=p[3], feats=p[5], sentence=sent,
                                 treebank=os.path.basename(f)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for lang in sys.argv[1:]:
        df = build(lang)
        df.to_parquet(f"{OUT}/{lang}.parquet", index=False)
        s = df.drop_duplicates("form").split.value_counts().to_dict()
        print(lang, "forms", s, "analyses", len(df), "rules", df.rule.nunique(),
              "top200 rule cov", round(df.rule.value_counts().head(200).sum() / len(df), 3),
              "contlex cov", round((df.contlex != "").mean(), 3), flush=True)
        u = ud_oov(lang)
        if u is not None and len(u):
            u.to_parquet(f"{OUT}/{lang}.ud_oov.parquet", index=False)
            print("   UD test OOV tokens:", len(u), u.upos.value_counts().head(5).to_dict(), flush=True)
        u = ud_oov(lang, "train")
        if u is not None and len(u):
            u.to_parquet(f"{OUT}/{lang}.ud_oov_train.parquet", index=False)
