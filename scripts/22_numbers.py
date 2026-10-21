"""Collect every number cited in the paper text into work/runs/numbers.json (used for writing and fact-checking)."""
import glob, json, os
import numpy as np

W = os.environ.get("LEXIBERT_WORK", "work") + "/runs"
ENC = ["mmbert_base", "pt_mlm", "pt_fst", "pt_full", "pt_align", "pt_gloss"]
LEX = "pt_gloss"  # LexiBERT
ME = ["me_mmbert_base", "me_pt_full", "me_pt_full_nohyp", f"me_{LEX}", f"me_{LEX}_nohyp"]
load = lambda p: json.load(open(p)) if os.path.exists(p) else None
N = {}


def mono(n, tb, m):
    v = [json.load(open(f))["results"][tb][m] for f in glob.glob(f"{W}/ud/{n}_{tb}_mono_s*.json")]
    return round(float(np.mean(v)), 1) if v else None


st = load(f"{W}/ud/stanza_released.json")["results"]
for tb in ("myv", "sme"):
    for m in ("upos", "feats", "lemma", "lemma_fst", "uas", "las"):
        N[f"mono/stanza/{tb}/{m}"] = round(st[tb][m], 1) if m in st[tb] else None
        for n in ENC:
            N[f"mono/{n}/{tb}/{m}"] = mono(n, tb, m)
for n in ENC:
    r = load(f"{W}/ud/{n}_multi_s1.json")
    if r:
        for m in ("upos", "feats", "lemma", "lemma_fst", "las"):
            N[f"multi/{n}/avg/{m}"] = round(float(np.mean([v[m] for v in r["results"].values()])), 1)
        for tb, v in r["results"].items():
            for m in ("upos", "lemma", "lemma_fst", "las"):
                N[f"multi/{n}/{tb}/{m}"] = v[m]
oov_l = ["myv", "mdf", "kpv", "koi", "udm", "mhr", "mrj", "sme", "smn", "sms", "olo", "krl", "vep", "liv", "yrk"]
for n in ENC:
    o = load(f"{W}/oov/{n}.json")
    if o:
        for m in ("lemma@1", "lemma@5", "full@1", "contlex@1", "contlex@3"):
            N[f"oov/{n}/avg/{m}"] = round(float(np.mean([o[l][m] for l in oov_l if m in o.get(l, {})])), 1)
        for k, v in o.get("strata", {}).items():
            for m, x in v.items():
                N[f"oov/{n}/strata/{k}/{m}"] = x
        for l in oov_l:
            for m, x in o.get(l, {}).items():
                N[f"oov/{n}/{l}/{m}"] = x
        for k, v in o.items():
            if k.endswith("_ud_oov"):
                for m, x in v.items():
                    N[f"oov/{n}/{k}/{m}"] = x
for n in ENC + ME:
    d = load(f"{W}/downstream/{n}.json")
    if not d:
        continue
    if "sentiment" in d:
        for l in ("eng(sst2-dev)", "fin", "kpv", "mdf", "myv", "udm"):
            vals = [d["sentiment"][s][l] for s in d["sentiment"]]
            N[f"sent/{n}/{l}"] = round(float(np.mean(vals)), 1); N[f"sent/{n}/{l}/sd"] = round(float(np.std(vals)), 1)
    for k in ("sentiment_static", "sentiment_static_aligned"):
        for l, x in d.get(k, {}).items():
            N[f"{k}/{n}/{l}"] = x
    for k in ("retrieval", "retrieval_static"):
        for l, x in d.get(k, {}).items():
            N[f"{k}/{n}/{l}"] = x
for vs in ("none", "stanza", "a2023", "fasttext") + tuple(ME):
    for tb in ("myv", "sme"):
        fs = glob.glob(f"{W}/ud_static/{vs}_{tb}_mono_s*.json")
        if fs:
            for m in ("upos", "feats", "las", "vec_coverage"):
                N[f"static/{vs}/{tb}/{m}"] = round(float(np.mean([json.load(open(f))["results"][tb][m] for f in fs])), 1)
            N[f"static/{vs}/{tb}/n"] = len(fs)
    r = load(f"{W}/ud_static/{vs}_multi_s1.json")
    if r:
        for m in ("upos", "las"):
            N[f"static/{vs}/multi_avg/{m}"] = round(float(np.mean([v[m] for v in r["results"].values()])), 1)
g, e = load(f"{W}/linkpred_graph.json"), load(f"{W}/linkpred_me_{LEX}.json")
for k in g:
    N[f"lp/{k}/jaccard"] = g[k]["jaccard"]["hits@1"]; N[f"lp/{k}/ranker"] = g[k]["ours"]["hits@1"]
    N[f"lp/{k}/recall_graph"] = round(100 * g[k]["cand_recall_graph"], 1)
    if e and k in e:
        N[f"lp/{k}/ranker_emb"] = e[k]["ours"]["hits@1"]; N[f"lp/{k}/recall_emb"] = round(100 * e[k]["cand_recall"], 1)
        N[f"lp/{k}/bli_p1"] = e[k]["bli_emb_only"]["p@1"]
for m in ("jaccard", "ranker", "ranker_emb"):
    vals = [N[f"lp/{k}/{m}"] for k in g if f"lp/{k}/{m}" in N]
    N[f"lp/avg/{m}"] = round(float(np.mean(vals)), 1)
for n in ENC:
    r = load(f"{W}/autocorrect/{n}.json")
    if r:
        L = [l for l in r if l != "weights"]
        for s in ("freq_lexicon", "lm_lexicon", "lm_fst"):
            for c in ("diacritics", "typo", "clean"):
                N[f"ac/{n}/{s}/{c}"] = round(float(np.mean([r[l][s][c] for l in L])), 1)
            bf = {}
            for l in L:
                for k, v in r[l][s]["by_freq"].items():
                    bf.setdefault(k, []).append(v)
            for k, v in bf.items():
                N[f"ac/{n}/{s}/by_freq/{k}"] = round(float(np.mean(v)), 1)
# post-submission: FST-analysis decoding (24), glossed sentiment (26), LLM reranking (25)
WJ = W.replace("/runs", "/runs_joint/ud")
for n in ENC:
    for tb in ("myv", "sme"):
        fs = glob.glob(f"{WJ}/{n}_{tb}_mono_s*.json")
        for m in ("free_upos", "free_feats", "free_lemma", "joint_upos", "joint_feats", "joint_lemma", "free_all", "joint_all"):
            if fs:
                N[f"joint/{n}/{tb}/{m}"] = round(float(np.mean([json.load(open(f))["results"][tb][m] for f in fs])), 1)
    r = load(f"{WJ}/{n}_multi_s1.json")
    if r:
        for m in ("free_upos", "free_feats", "free_lemma", "joint_upos", "joint_feats", "joint_lemma", "free_all", "joint_all"):
            N[f"joint/{n}/multi_avg/{m}"] = round(float(np.mean([v[m] for v in r["results"].values()])), 1)
        for tb, v in r["results"].items():
            for m in ("free_upos", "free_feats", "joint_upos", "joint_feats"):
                N[f"joint/{n}/multi/{tb}/{m}"] = v[m]
WG = W.replace("/runs", "/runs_gloss")
for f in ("sentiment.json", "sentiment_pt_gloss_pt_align.json"):
    for n, r in (load(f"{WG}/{f}") or {}).get("mean", {}).items():
        for l, v in r.items():
            for k, x in v.items():
                N[f"gloss_sent/{n}/{l}/{k}"] = x
WL = W.replace("/runs", "/runs_llm")
for tag in ("", "_gloss"):
    r = load(f"{WL}/rerank_qwen3.8-27b_test{tag}.json")
    if r:
        for k, v in r["results"].items():
            for m, x in v.items():
                N[f"llm{tag}/{k}/{m}"] = x
json.dump(N, open(f"{W}/numbers.json", "w"), indent=0, ensure_ascii=False, sort_keys=True)
print(len(N), "numbers collected")
