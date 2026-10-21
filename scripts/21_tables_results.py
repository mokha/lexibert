"""Generate all result tables (paper/tables/*.tex) from work/runs/**.json (and the post-submission runs_joint,
runs_gloss and runs_llm). Missing runs print as '--'."""
import glob, json, os
import numpy as np

W = os.environ.get("LEXIBERT_WORK", "work") + "/runs"
WJ = os.environ.get("LEXIBERT_WORK", "work") + "/runs_joint/ud"  # FST-analysis decoding of the saved parsers (24_fst_joint)
WG = os.environ.get("LEXIBERT_WORK", "work") + "/runs_gloss"  # dictionary-gloss sentiment (26_sentiment_gloss)
WL = os.environ.get("LEXIBERT_WORK", "work") + "/runs_llm"  # LLM reranking (25_llm_rerank)
OUT = os.environ.get("LEXIBERT_TABLES", "tables")
LEX = "pt_gloss"  # LexiBERT: masked language modelling + dictionary gloss pairs
ENCODERS = [("mmbert_base", "mmBERT-small"), ("pt_mlm", "+ MLM"), ("pt_fst", "+ MLM + FST"),
            ("pt_full", "+ MLM + FST + CS + pairs"), ("pt_align", "+ MLM + gloss alignment"), (LEX, "\\enc")]
MAIN = [("mmbert_base", "mmBERT-small"), ("pt_mlm", "+ MLM"), (LEX, "\\enc")]  # rows of the main result tables
TB_NAMES = {"myv": "Erzya", "mdf": "Moksha", "kpv_lattice": "Komi-Z. Lattice", "kpv_ikdp": "Komi-Z. IKDP", "koi": "Komi-Permyak",
            "sms": "Skolt Sámi", "olo": "Livvi", "krl": "Karelian", "sme": "North Sámi", "vep": "Veps", "yrk": "Nenets"}


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def seeds(pattern, tb, metric):
    vals = []
    for f in sorted(glob.glob(pattern)):
        r = load(f)
        if r and tb in r["results"] and metric in r["results"][tb]:
            vals.append(r["results"][tb][metric])
    return vals


def fmt(vals, sd=True, bold=False):
    if not vals:
        return "--"
    m = np.mean(vals)
    s = f"{m:.1f}"
    if sd and len(vals) > 1:
        s += f"\\textsubscript{{±{np.std(vals):.1f}}}"
    return f"\\textbf{{{s}}}" if bold else s


def best_mask(cols):
    """Per column, which row has the highest mean (for bolding)."""
    means = [[np.mean(v) if v else -1 for v in row] for row in cols]
    best = np.argmax(np.array(means), axis=0)
    return best


def table_ud_mono():
    metrics = [("upos", "UPOS"), ("feats", "UFeats"), ("lemma", "Lem"), ("lemma_fst", "Lem$_{\\text{FST}}$"), ("uas", "UAS"), ("las", "LAS")]
    rows = []
    st = load(f"{W}/ud/stanza_released.json")
    rows.append(("Stanza (released)", [[st["results"][tb][m]] if st and m in st["results"].get(tb, {}) else []
                                       for tb in ("myv", "sme") for m, _ in metrics]))
    for key, name in MAIN:
        rows.append((name, [seeds(f"{W}/ud/{key}_{tb}_mono_s*.json", tb, m) for tb in ("myv", "sme") for m, _ in metrics]))
    joint = {"upos": "joint_upos", "feats": "joint_feats", "lemma_fst": "joint_lemma"}  # same trees, so UAS/LAS as above
    rows.append(("\\quad + FST analyses", [seeds(f"{WJ}/{LEX}_{tb}_mono_s*.json", tb, joint[m]) if m in joint else []
                                             for tb in ("myv", "sme") for m, _ in metrics]))
    best = best_mask([r[1] for r in rows])
    lines = ["\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{3.5pt}",
             "\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}{l" + "r" * 12 + "}\\toprule",
             " & \\multicolumn{6}{c}{Erzya JR} & \\multicolumn{6}{c}{North Sámi Giella}\\\\",
             "\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}",
             "Model & " + " & ".join(n for _, n in metrics) * 1 + " & " + " & ".join(n for _, n in metrics) + "\\\\\\midrule"]
    for i, (name, cols) in enumerate(rows):
        lines.append(name + " & " + " & ".join(fmt(c, bold=(best[j] == i)) for j, c in enumerate(cols)) + "\\\\")
        if i == 0:
            lines.append("\\midrule")
    lines += ["\\bottomrule\\end{tabular}\\end{adjustbox}",
              "\\caption{UD results when training on the treebank itself (gold tokenisation; mean and standard deviation "
              "over 3 seeds). Lem$_{\\text{FST}}$: lemma restricted to the FST's candidate lemmas when the FST knows the word; "
              "+ FST analyses: UPOS, UFeats and lemma chosen jointly among the FST's analyses (the trees do not change). "
              "Stanza's released models are run on the same gold tokenisation.}\\label{tab:ud-mono}\\end{table*}"]
    open(f"{OUT}/ud_mono.tex", "w").write("\n".join(lines) + "\n")


def table_ud_multi():
    tbs = [tb for tb in TB_NAMES if tb not in ("myv", "sme")] + ["myv", "sme"]
    cols = [("upos", "UPOS"), ("lemma_fst", "Lem$_{\\text{FST}}$"), ("las", "LAS")]
    data = {k: load(f"{W}/ud/{k}_multi_s1.json") for k, _ in ENCODERS}
    lines = ["\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}{l" + "r" * 6 + "}\\toprule",
             " & \\multicolumn{3}{c}{mmBERT-small} & \\multicolumn{3}{c}{\\enc}\\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Treebank & " + " & ".join(n for _, n in cols) * 1 + " & " + " & ".join(n for _, n in cols) + "\\\\\\midrule"]
    avg = {k: {m: [] for m, _ in cols} for k, _ in ENCODERS}
    for tb in tbs:
        cells = []
        for k in ("mmbert_base", LEX):
            for m, _ in cols:
                v = data[k]["results"].get(tb, {}).get(m) if data[k] else None
                cells.append(v)
        for k, _ in ENCODERS:
            for m, _ in cols:
                v = data[k]["results"].get(tb, {}).get(m) if data[k] else None
                if v is not None:
                    avg[k][m].append(v)
        b = [cells[i] if cells[i] is not None else -1 for i in range(6)]
        out = []
        for i, c in enumerate(cells):
            other = b[i + 3] if i < 3 else b[i - 3]
            out.append("--" if c is None else (f"\\textbf{{{c:.1f}}}" if c > other else f"{c:.1f}"))
        lines.append(TB_NAMES[tb] + " & " + " & ".join(out) + "\\\\")
    lines.append("\\midrule")
    for k, name in MAIN:
        lines.append(f"\\multicolumn{{1}}{{l}}{{Avg. {name}}} & \\multicolumn{{6}}{{c}}{{" +
                     " / ".join(f"{np.mean(avg[k][m]):.1f}" if avg[k][m] else "--" for m, _ in cols) + "}\\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{adjustbox}",
              "\\caption{One multilingual model trained on all Uralic training splits plus Finnish and Estonian, tested on every "
              "Uralic treebank (most are test-only). The average rows give UPOS / Lem$_{\\text{FST}}$ / LAS over all 11 test "
              "sets for each pretraining variant.}\\label{tab:ud-multi}\\end{table}"]
    open(f"{OUT}/ud_multi.tex", "w").write("\n".join(lines) + "\n")


def table_static():
    sets = [("none", "None"), ("stanza", "Stanza default"), ("a2023", "\\citet{alnajjar-etal-2023-sentiment}"),
            ("fasttext", "fastText (our corpus)"), ("me_mmbert_base", "\\me\\ $\\leftarrow$ mmBERT"),
            (f"me_{LEX}_nohyp", "\\me\\ $\\leftarrow$ \\enc"), (f"me_{LEX}", "\\quad + unknown words")]
    unavailable = {("a2023", "sme")}  # the 2023 vectors exist only for Erzya, Moksha, Komi-Zyrian and Udmurt
    rows = []
    for key, name in sets:
        cells = []
        for tb in ("myv", "sme"):
            las = [] if (key, tb) in unavailable else seeds(f"{W}/ud_static/{key}_{tb}_mono_s*.json", tb, "las")
            cov = [] if (key, tb) in unavailable else seeds(f"{W}/ud_static/{key}_{tb}_mono_s*.json", tb, "vec_coverage")
            cells += [las, cov]
        r = load(f"{W}/ud_static/{key}_multi_s1.json")
        cells.append([np.mean([v["las"] for v in r["results"].values()])] if r else [])
        rows.append((name, cells))
    best = best_mask([[c if j % 2 == 0 else [] for j, c in enumerate(r[1])] for r in rows])  # LAS columns only
    lines = ["\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}",
             "\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}{lrrrrr}\\toprule",
             " & \\multicolumn{2}{c}{Erzya} & \\multicolumn{2}{c}{North Sámi} & All\\\\\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-6}",
             "Vectors & LAS & Cov. & LAS & Cov. & LAS\\\\\\midrule"]
    for i2, (name, cells) in enumerate(rows):
        lines.append(name + " & " + " & ".join(fmt(c, sd=(j % 2 == 0), bold=(best[j] == i2 and j % 2 == 0)) for j, c in enumerate(cells)) + "\\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{adjustbox}",
              "\\caption{LAS of the same BiLSTM parser with different frozen word vectors (3 seeds) and the share of test "
              "words that get a vector (Cov.). All: average LAS of one multilingual parser over the 11 Uralic test sets.}\\label{tab:static}\\end{table}"]
    open(f"{OUT}/static.tex", "w").write("\n".join(lines) + "\n")


def table_oov():
    langs = ["myv", "mdf", "kpv", "koi", "udm", "mhr", "mrj", "sme", "smn", "sms", "olo", "krl", "vep", "liv", "yrk"]
    o = load(f"{W}/oov/{LEX}.json") or {}
    cols = [("lemma@1", "L@1"), ("lemma@5", "L@5"), ("contlex@1", "C@1"), ("contlex@3", "C@3")]
    half = (len(langs) + 1) // 2
    lines = ["\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}{lrrrr@{\\hspace{8pt}}lrrrr}\\toprule",
             " & " + " & ".join(n for _, n in cols) + " & & " + " & ".join(n for _, n in cols) + "\\\\\\midrule"]
    cell = lambda l: " & ".join("--" if o.get(l, {}).get(m) is None else f"{o[l][m]:.1f}" for m, _ in cols)
    for k in range(half):
        left = langs[k]; right = langs[k + half] if k + half < len(langs) else None
        lines.append(f"{left} & {cell(left)} & " + (f"{right} & {cell(right)}" if right else " & & & & ") + "\\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{adjustbox}",
              "\\caption{Unknown word analysis with \\enc\\ on held-out lemmas: the correct lemma among the first (L@1) or "
              "five best (L@5) suggestions, and the correct inflection class among the first (C@1) or three best (C@3).}"
              "\\label{tab:oov}\\end{table}"]
    open(f"{OUT}/oov.tex", "w").write("\n".join(lines) + "\n")


def table_linkpred():
    g, e = load(f"{W}/linkpred_graph.json"), load(f"{W}/linkpred_me_{LEX}.json")
    llm = (load(f"{WL}/rerank_qwen3.8-27b_test_gloss.json") or {}).get("results", {})
    keys = sorted((g or {}).keys(), key=lambda k: (k.split("-")[1], k))
    rows = [("Hits@1: Jaccard", lambda a, b: a["jaccard"]["hits@1"]), ("\\quad Ranker", lambda a, b: a["ours"]["hits@1"]),
            ("\\quad + \\me", lambda a, b: b["ours"]["hits@1"] if b else None),
            ("\\quad + LLM", lambda a, b: None)]  # filled from the LLM results below
    vals = [[f(g[k], (e or {}).get(k)) for k in keys] for _, f in rows]
    vals[3] = [llm.get(k, {}).get("llm_free_hits@1") for k in keys]
    vals = [v + [np.mean(v) if None not in v else None] for v in vals]
    tgts = [k.split("-")[1] for k in keys]
    n_eng = tgts.count("eng")
    lines = ["\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{2.4pt}",
             "\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}{l" + "r" * (len(keys) + 1) + "}\\toprule",
             f" & \\multicolumn{{{n_eng}}}{{c}}{{$\\to$ English}} & \\multicolumn{{{len(keys) - n_eng}}}{{c}}{{$\\to$ Finnish}} & \\\\"
             f"\\cmidrule(lr){{2-{n_eng + 1}}}\\cmidrule(lr){{{n_eng + 2}-{len(keys) + 1}}}",
             " & " + " & ".join(k.split("-")[0] for k in keys) + " & Avg.\\\\\\midrule"]
    for r, ((name, _), v) in enumerate(zip(rows, vals)):
        out = []
        for c, x in enumerate(v):
            top = max((vals[i][c] for i in range(4) if vals[i][c] is not None), default=None)
            out.append("--" if x is None else (f"\\textbf{{{x:.1f}}}" if r < 4 and x == top else f"{x:.1f}"))
        lines.append(name + " & " + " & ".join(out) + "\\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{adjustbox}",
              "\\caption{Translation prediction for held-out Ve'rdd words. Hits@1 of the Jaccard coefficient over the "
              "translation graph, our ranker over the graph candidates, the ranker with \\me\\ candidates and the LLM reranking "
              "of its ten best candidates.}\\label{tab:linkpred}\\end{table*}"]
    open(f"{OUT}/linkpred.tex", "w").write("\n".join(lines) + "\n")


def table_downstream():
    langs = ["fin", "kpv", "mdf", "myv", "udm"]
    prior = {"fin": 75, "kpv": 56, "mdf": 63, "myv": 69, "udm": 63}
    lines = ["\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{2.5pt}", "\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}{lrrrrrrr}\\toprule",
             " & \\multicolumn{5}{c}{Sentiment acc. (train: English)} & \\multicolumn{2}{c}{kpv--rus P@1}\\\\\\cmidrule(lr){2-6}\\cmidrule(lr){7-8}",
             "Model & " + " & ".join(langs) + " & $\\to$ & $\\leftarrow$\\\\\\midrule",
             "Alnajjar et al. (2023) & " + " & ".join(str(prior[l]) for l in langs) + " & -- & --\\\\\\midrule"]
    for k, name in [("mmbert_base", "mmBERT-small"), (LEX, "\\enc")]:
        r = load(f"{W}/downstream/{k}.json")
        if not r:
            lines.append(name + " & " + " & ".join(["--"] * 7) + "\\\\"); continue
        sent = r.get("sentiment", {})
        cells = [fmt([s[l] for s in sent.values()]) if sent else "--" for l in langs]
        ret = r.get("retrieval", {})
        lines.append(name + " & " + " & ".join(cells) + f" & {ret.get('kpv->rus', '--')} & {ret.get('rus->kpv', '--')}\\\\")
    lines.append("\\midrule")
    gl = {**((load(f"{WG}/sentiment.json") or {}).get("per_seed", {})),
          **((load(f"{WG}/sentiment_pt_gloss_pt_align.json") or {}).get("per_seed", {}))}
    for k, name in [("mmbert_base", "mmBERT-small, glossed"), (LEX, "\\enc, glossed")]:
        ps = gl.get(k, {})
        lines.append(name + " & " + " & ".join(fmt([ps[sd][l]["gloss"] for sd in ps]) if ps else "--" for l in langs) + " & -- & --\\\\")
    lines.append("\\midrule")
    for k, name in [(f"me_{LEX}", "\\me\\ $\\leftarrow$ \\enc")]:
        r = load(f"{W}/downstream/{k}.json")
        if not r:
            lines.append(name + " & " + " & ".join(["--"] * 7) + "\\\\"); continue
        s, ret = r.get("sentiment_static", {}), r.get("retrieval_static", {})
        lines.append(name + " & " + " & ".join(f"{s[l]:.1f}" if l in s else "--" for l in langs) +
                     f" & {ret.get('kpv->rus', '--')} & {ret.get('rus->kpv', '--')}\\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{adjustbox}",
              "\\caption{Zero-shot transfer. Sentiment: trained on English SST-2 only, tested on the 68-sentence aligned set "
              "(encoders: 3 seeds); glossed: the test sentences are replaced by English glosses through FST lemmas and Ve'rdd. "
              "Retrieval: P@1 over 500 held-out Komi-Zyrian--Russian example pairs, no training.}"
              "\\label{tab:downstream}\\end{table}"]
    open(f"{OUT}/downstream.tex", "w").write("\n".join(lines) + "\n")


def table_autocorrect():
    """Compact: averages over the 10 languages per system and error type, plus words unseen in the corpus."""
    rows = [("Keep the typed word", "mmbert_base", "keep_typed"), ("Frequency", "mmbert_base", "freq_lexicon"),
            ("mmBERT-small", "mmbert_base", "lm_lexicon"), ("\\quad + FST candidates", "mmbert_base", "lm_fst"),
            ("\\enc", LEX, "lm_lexicon"), ("\\quad + FST candidates", LEX, "lm_fst")]
    runs = {k: load(f"{W}/autocorrect/{k}.json") for k in ("mmbert_base", LEX)}
    lines = ["\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}",
             "\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}{lrrrr}\\toprule",
             "System & Missing letters & Typos & Correct & Unseen\\\\\\midrule"]
    table = []
    for name, run, sysn in rows:
        r = runs.get(run)
        if not r:
            table.append((name, [None] * 4)); continue
        L = [l for l in r if l != "weights"]
        cells = [np.mean([r[l][sysn][c] for l in L if c in r[l][sysn]]) for c in ("diacritics", "typo", "clean")]
        unseen = [v for l in L for k, v in r[l][sysn].get("by_freq", {}).items() if k.endswith("/unseen") and not k.startswith("clean")]
        cells.append(np.mean(unseen) if unseen else (0.0 if sysn == "keep_typed" else None))
        table.append((name, cells))
    best = [max(c[j] for _, c in table[1:] if c[j] is not None) for j in range(4)]
    for k, (name, cells) in enumerate(table):
        lines.append(name + " & " + " & ".join("--" if c is None else (f"\\textbf{{{c:.1f}}}" if k > 0 and c == best[j] else f"{c:.1f}")
                                               for j, c in enumerate(cells)) + "\\\\")
        if k == 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule\\end{tabular}\\end{adjustbox}",
              "\\caption{Keyboard autocorrection accuracy, averaged over 10 languages: words with missing letters, words with "
              "a typo, correct words that should be left unchanged, and corrupted words that never occur in our corpus.}"
              "\\label{tab:autocorrect}\\end{table}"]
    open(f"{OUT}/autocorrect.tex", "w").write("\n".join(lines) + "\n")


def table_ablation():
    oov_langs = ["myv", "mdf", "kpv", "koi", "udm", "mhr", "mrj", "sme", "smn", "sms", "olo", "krl", "vep", "liv", "yrk"]
    cols = ["Erzya", "N.Sámi", "UPOS", "LAS", "L@1", "C@1", "Sent.", "Retr."]
    lines = ["\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}{l" + "r" * len(cols) + "}\\toprule",
             " & \\multicolumn{2}{c}{LAS (mono)} & \\multicolumn{2}{c}{Multi, avg.} & \\multicolumn{2}{c}{Unknown} & \\multicolumn{2}{c}{Transfer}\\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
             "Model & " + " & ".join(cols) + "\\\\\\midrule"]
    rows = []
    for key, name in ENCODERS:
        cells = [np.mean(v) if (v := seeds(f"{W}/ud/{key}_{tb}_mono_s*.json", tb, "las")) else None for tb in ("myv", "sme")]
        m = load(f"{W}/ud/{key}_multi_s1.json")
        cells += [np.mean([x[k] for x in m["results"].values()]) if m else None for k in ("upos", "las")]
        o = load(f"{W}/oov/{key}.json")
        cells += [np.mean([o[l][k] for l in oov_langs if k in o.get(l, {})]) if o else None for k in ("lemma@1", "contlex@1")]
        d = load(f"{W}/downstream/{key}.json")
        if d and d.get("sentiment"):
            sent = d["sentiment"]
            cells.append(np.mean([np.mean([sent[sd][l] for sd in sent]) for l in ("kpv", "mdf", "myv", "udm")]))
        else:
            cells.append(None)
        cells.append(np.mean(list(d["retrieval"].values())) if d and d.get("retrieval") else None)
        rows.append((name, cells))
    best = [max((r[1][j] for r in rows if r[1][j] is not None), default=None) for j in range(len(cols))]
    for name, cells in rows:
        lines.append(name + " & " + " & ".join("--" if c is None else (f"\\textbf{{{c:.1f}}}" if c == best[j] else f"{c:.1f}")
                                               for j, c in enumerate(cells)) + "\\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{adjustbox}",
              "\\caption{Ablation of the pretraining objectives (CS: dictionary code-switching; pairs: translation pairs). LAS when training on Erzya or North Sámi (3 seeds); average "
              "UPOS and LAS of the multilingual model over the 11 Uralic test sets; average accuracy of the first lemma (L@1) and "
              "inflection class (C@1) for unknown words; average sentiment accuracy over Erzya, Moksha, Komi-Zyrian and Udmurt "
              "with English training data; and Komi-Zyrian--Russian retrieval P@1 (mean of both directions).}"
              "\\label{tab:ablation}\\end{table}"]
    open(f"{OUT}/ablation.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (table_ud_mono, table_ud_multi, table_static, table_oov, table_linkpred, table_downstream, table_autocorrect, table_ablation):
        try:
            f()
            print("ok", f.__name__)
        except Exception as ex:
            print("FAIL", f.__name__, repr(ex))
