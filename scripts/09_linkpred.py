"""Translation link prediction on the Ve'rdd graph.

For each Uralic source language L and target T (eng, fin): hold out 10% of live L-T translation
edges for test and 10% for ranker training, remove both from the graph, then rank every
T-lexeme reachable within two hops (through any pivot language) for each held-out source lexeme.

Baseline: Jaccard coefficient over the pivot graph (the method in Ve'rdd's predict_translations).
Ours:     LightGBM lambdarank over graph, pivot-diversity, POS and string features.
Only the current Ve'rdd relations are used (relations.parquet)."""
import collections, json, os, random, sys
import numpy as np, pandas as pd
import lightgbm as lgb
import torch
from rapidfuzz.distance import Levenshtein
from unidecode import unidecode

W = os.environ.get("LEXIBERT_WORK", "work")
SRC = ["myv", "mdf", "kpv", "koi", "udm", "sme", "sms", "smn", "olo", "krl", "liv", "mhr", "mrj"]
TGT = ["eng", "fin"]
POS_MAP = {"n": "N", "adj": "A", "vblex": "V", "adv": "Adv", "np": "N"}


def load():
    lx = pd.read_parquet(f"{W}/data/verdd/lexemes.parquet", columns=["id", "lang", "lexeme", "pos"])
    rel = pd.read_parquet(f"{W}/data/verdd/relations.parquet", columns=["type", "f", "t", "checked"])
    rel = rel[(rel.type == 0) & rel.f.isin(lx.id) & rel.t.isin(lx.id) & (rel.f != rel.t)]
    return lx.set_index("id"), rel


def features(s, c, adj, lang, text, pos, deg):
    ns, nc = adj[s], adj[c]
    common = ns & nc
    union = len(ns | nc)
    aa = sum(1 / np.log(max(deg[z], 2)) for z in common)
    ra = sum(1 / max(deg[z], 1) for z in common)
    pivots = {lang[z] for z in common}
    a, b = unidecode(text[s]).lower(), unidecode(text[c]).lower()
    return [len(common) / union if union else 0, len(common), aa, ra, len(pivots), int("fin" in pivots),
            int("rus" in pivots), int("eng" in pivots), deg[s], deg[c], int(bool(pos[s]) and pos[s] == pos[c]),
            Levenshtein.normalized_similarity(a, b), abs(len(a) - len(b)) / max(len(a), len(b), 1),
            int(a[:3] == b[:3])]


DUMP = []
RUNS = os.environ.get("RUNS", "runs")
NAME = "graph"
EMB = None  # (model, featurizer) of a morph_embeddings student when --emb is given
ALIGN = False
K_EMB = 20


def emb_vectors(ids, lx):
    me = __import__("importlib").import_module("11_morph_embeddings")
    model, feat = EMB
    items = [(lx.lang[i], lx.lexeme[i]) for i in ids]
    return torch.nn.functional.normalize(me.embed(model, feat, items, "cuda"), dim=-1)


FEATS = ["jaccard", "common", "adamic_adar", "resource_alloc", "n_pivot_langs", "via_fin", "via_rus", "via_eng",
         "deg_src", "deg_cand", "pos_match", "str_sim", "len_diff", "prefix3"]


def run(L, T, lx, rel, rng):
    lang, text = lx.lang.to_dict(), lx.lexeme.to_dict()
    pos = {i: (POS_MAP.get(p, p) if p else "") for i, p in lx.pos.items()}
    is_pair = ((rel.f.map(lang) == L) & (rel.t.map(lang) == T)) | ((rel.f.map(lang) == T) & (rel.t.map(lang) == L))
    pair = rel[is_pair]
    srcs = sorted({x for x in set(pair.f) | set(pair.t) if lang[x] == L})
    # stratified by part of speech: 10% of each POS for testing and 10% for training the ranker (at least 50 test
    # words per pair in total, so small pairs get a larger share)
    frac = max(0.1, 50 / max(1, len(srcs)))
    strata = collections.defaultdict(list)
    for x in srcs:
        strata[pos[x]].append(x)
    test_src, train_src = set(), set()
    for k in sorted(strata):
        g = strata[k]
        rng.shuffle(g)
        n = round(frac * len(g))
        test_src |= set(g[:n]); train_src |= set(g[n:2 * n])
    held = pair[pair.f.isin(test_src | train_src) | pair.t.isin(test_src | train_src)]
    gold = collections.defaultdict(set)
    for f, t in zip(held.f, held.t):
        s, g = (f, t) if lang[f] == L else (t, f)
        gold[s].add(g)
    graph = rel.drop(held.index)
    adj = collections.defaultdict(set)
    for f, t in zip(graph.f, graph.t):
        adj[f].add(t); adj[t].add(f)
    deg = collections.Counter({k: len(v) for k, v in adj.items()})

    def graph_cands(s):
        out = set()
        for z in adj[s]:
            out |= {c for c in adj[z] if lang.get(c) == T}
        return out

    emb_c, cos = {}, {}
    if EMB is not None:
        tgt_ids = lx.index[lx.lang == T].tolist()
        tv = emb_vectors(tgt_ids, lx).cuda()
        qs = sorted(test_src | train_src)
        qv = emb_vectors(qs, lx).cuda()
        if ALIGN:  # supervised orthogonal map L -> T from the graph's remaining (non-held-out) translations
            pairs = [(f, t) if lang[f] == L else (t, f) for f, t in zip(graph.f, graph.t)
                     if {lang.get(f), lang.get(t)} == {L, T}][:20000]
            if len(pairs) >= 50:
                xs = emb_vectors([a for a, _ in pairs], lx).cuda(); ys = emb_vectors([b for _, b in pairs], lx).cuda()
                u, _, vt = torch.linalg.svd(xs.T @ ys)
                qv = torch.nn.functional.normalize(qv @ (u @ vt), dim=-1)
                print(f"  aligned {L}->{T} with {len(pairs)} pairs", flush=True)
        for i in range(0, len(qs), 512):
            sim = qv[i:i + 512] @ tv.T
            top = sim.topk(K_EMB, dim=1)
            for r, s in enumerate(qs[i:i + 512]):
                emb_c[s] = {tgt_ids[j]: v for v, j in zip(top.values[r].tolist(), top.indices[r].tolist())}
                for g in gold[s] | graph_cands(s):
                    if g in lx.index:
                        cos[(s, g)] = None  # filled lazily below
        need = sorted({g for (_, g) in cos})
        if need:
            nv = emb_vectors(need, lx).cuda()
            pos_of = {g: k for k, g in enumerate(need)}
            for (s, g) in list(cos):
                cos[(s, g)] = float(qv[qs.index(s)] @ nv[pos_of[g]]) if s in qs else 0.0

    def cands(s, use_emb=True):
        out = graph_cands(s)
        if use_emb and s in emb_c:
            out |= set(emb_c[s])
        return out

    def table(sources):
        X, y, grp, meta = [], [], [], []
        for s in sorted(sources):
            cs = sorted(cands(s))
            if not cs:
                continue
            for c in cs:
                f = features(s, c, adj, lang, text, pos, deg)
                if EMB is not None:
                    e = emb_c.get(s, {})
                    f += [e.get(c, cos.get((s, c), 0.0) or 0.0), int(c in e)]
                X.append(f); y.append(int(c in gold[s])); meta.append((s, c))
            grp.append(len(cs))
        return np.array(X, dtype=float), np.array(y), grp, meta

    Xtr, ytr, gtr, _ = table(train_src)
    Xte, yte, gte, mte = table(test_src)
    if len(gtr) < 10 or len(gte) < 10:
        return None
    ranker = lgb.LGBMRanker(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=10, verbose=-1)
    ranker.fit(Xtr, ytr, group=gtr)
    os.makedirs(f"{W}/checkpoints/linkpred/{NAME}", exist_ok=True)
    ranker.booster_.save_model(f"{W}/checkpoints/linkpred/{NAME}/{L}-{T}.txt")
    scores = {"jaccard": Xte[:, 0], "ours": ranker.predict(Xte)}
    res = {"n_test_src": len(test_src), "covered_src": len(gte),
           "cand_recall_graph": round(sum(len(gold[s] & graph_cands(s)) > 0 for s in test_src) / len(test_src), 3),
           "cand_recall": round(sum(len(gold[s] & cands(s)) > 0 for s in test_src) / len(test_src), 3)}
    if EMB is not None:  # pure BLI: rank by embedding similarity only
        bli1 = sum(bool(emb_c.get(s)) and max(emb_c[s], key=emb_c[s].get) in gold[s] for s in test_src)
        bli10 = sum(bool(set(emb_c.get(s, {})) & gold[s]) for s in test_src)
        res["bli_emb_only"] = {"p@1": round(100 * bli1 / len(test_src), 1), "p@20": round(100 * bli10 / len(test_src), 1)}
    for name, sc in scores.items():
        h1 = h10 = mrr = 0
        i = 0
        for g in gte:
            order = np.argsort(-sc[i:i + g], kind="stable")
            ranks = np.where(yte[i:i + g][order] == 1)[0]
            if len(ranks):
                r = ranks[0] + 1
                h1 += r == 1; h10 += r <= 10; mrr += 1 / r
            i += g
        # metrics over all test sources (uncovered sources count as misses)
        res[name] = {"hits@1": round(100 * h1 / len(test_src), 1), "hits@10": round(100 * h10 / len(test_src), 1),
                     "mrr": round(100 * mrr / len(test_src), 1)}
    if os.environ.get("LP_DUMP"):  # examples for the paper: top-1 of Jaccard and of the ranker per test word
        i = 0
        srcs_te = [s for s in sorted(test_src) if cands(s)]
        for s, g in zip(srcs_te, gte):
            cs = [c for _, c in mte[i:i + g]]
            top = [cs[k] for k in np.argsort(-scores["ours"][i:i + g], kind="stable")[:10]]
            DUMP.append(dict(pair=f"{L}-{T}", src=text[s], gold=sorted(text[x] for x in gold[s]),
                             jaccard=text[cs[int(np.argmax(scores["jaccard"][i:i + g]))]],
                             ours=text[cs[int(np.argmax(scores["ours"][i:i + g]))]],
                             in_graph=bool(gold[s] & graph_cands(s)), id=int(s), pos=pos[s],
                             gold_ids=sorted(int(x) for x in gold[s]), n_test_src=len(test_src),
                             top=[(int(c), text[c]) for c in top],
                             known=sorted((lang[z], text[z]) for z in adj[s] if z in lang), split="test"))
            i += g
        # dev words for choosing how to use the dump (e.g. by an LLM) without the test set: a second ranker is
        # trained on half of the ranker's training words and ranks the other half
        tr = sorted(train_src)
        random.Random(1).shuffle(tr)
        half_a, half_b = set(tr[:len(tr) // 2]), set(tr[len(tr) // 2:])
        Xa, ya, ga, _ = table(half_a)
        Xb, yb, gb, mb = table(half_b)
        if len(ga) >= 10 and gb:
            rk = lgb.LGBMRanker(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=10, verbose=-1)
            rk.fit(Xa, ya, group=ga)
            sb, i = rk.predict(Xb), 0
            for s, g in zip([s for s in sorted(half_b) if cands(s)], gb):
                cs = [c for _, c in mb[i:i + g]]
                top = [cs[k] for k in np.argsort(-sb[i:i + g], kind="stable")[:10]]
                DUMP.append(dict(pair=f"{L}-{T}", src=text[s], gold=sorted(text[x] for x in gold[s]), id=int(s),
                                 pos=pos[s], gold_ids=sorted(int(x) for x in gold[s]), n_test_src=len(half_b),
                                 top=[(int(c), text[c]) for c in top],
                                 known=sorted((lang[z], text[z]) for z in adj[s] if z in lang), split="dev"))
                i += g
    res["importance"] = dict(zip(FEATS + (["emb_cos", "emb_topk"] if EMB is not None else []), ranker.feature_importances_.tolist()))
    return res


def main():
    global EMB, ALIGN, NAME
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb")
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--name", default="graph")
    ap.add_argument("langs", nargs="*")
    a = ap.parse_args()
    NAME = a.name
    lx, rel = load()
    ALIGN = a.align
    if a.emb:
        sys.path.insert(0, os.path.dirname(__file__))
        me = __import__("importlib").import_module("11_morph_embeddings")
        EMB = me.load_student(f"{W}/emb/{a.emb}", SRC)
    out = {}
    for L in a.langs or SRC:
        for T in TGT:
            r = run(L, T, lx, rel, random.Random(0))
            if r:
                out[f"{L}-{T}"] = r
                print(L, T, {k: v for k, v in r.items() if k != "importance"}, flush=True)
    if os.environ.get("LP_DUMP"):
        json.dump(DUMP, open(os.environ["LP_DUMP"], "w"), ensure_ascii=False)
        return
    os.makedirs(f"{W}/{RUNS}", exist_ok=True)
    json.dump(out, open(f"{W}/{RUNS}/linkpred_{a.name}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
