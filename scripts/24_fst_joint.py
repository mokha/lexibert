"""Joint FST decoding for the saved UD parsers: the FST lists the possible analyses of a word, and the parser's UPOS,
UFeats and lemma scores pick one of them, instead of predicting the three independently.

FST tags are mapped to UD (UPOS, UFeats) with a table learned from the training treebanks of each model: for every
training word, each FST analysis whose lemma matches the gold lemma is paired with the gold UPOS and UFeats. At test
time, every analysis (lemma, tags) of a word gives the candidates (lemma, UPOS, UFeats) for all the UD pairs seen with
its tags, and the candidate with the highest summed log-probability wins. Words that the FST does not know, or whose
tags were never seen in training, keep the free predictions (with the FST lemma restriction of 07_ud).
  python scripts/24_fst_joint.py [checkpoint names...]      -> work/runs_joint/ud/*.json"""
import glob, json, os, sys
from collections import Counter, defaultdict
from importlib import import_module
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
ud = import_module("07_ud")
re = import_module("23_reeval")
W = ud.W
OUT = f"{W}/runs_joint/ud"
DROP = ("Sem/", "Err/", "Use/", "Hom", "Dyn", "Gram/", "Foc/", "Clt/")


def analyses(tb, form):
    """(lemma, tag key) pairs of a form; for compounds, the tags of the last part and the full lemma is not known."""
    lang = ud.FST_LANG.get(tb)
    if lang is None:
        return []
    ud.fst_lemmas(tb, form)  # loads the cache
    out = []
    for a in list(ud._FST[lang].get(form, [])) or list(ud._FST[lang].get(form.lower(), [])):
        compound = "#" in a
        last = a.split("#")[-1].split("+")
        key = "+".join(t for t in last[1:] if not t.startswith(DROP))
        out.append((None if compound else last[0], key))
    return out


def tag_table(train_tbs):
    table = defaultdict(Counter)
    for tb in train_tbs:
        for s in ud.split(tb, "train"):
            for t in s:
                an = [(l, k) for l, k in analyses(tb, t["form"]) if l is not None and l.lower() == t["lemma"].lower()]
                for _, k in set(an):
                    table[k][(t["upos"], t["feats"])] += 1
    return table


def decode(form, tb, table, lu, lf, ll, v, free):
    """Best (lemma, upos, feats) among the FST candidates, or None when there are none."""
    best = None
    for lemma, key in set(analyses(tb, form)):
        if key not in table:
            continue
        if lemma is None:
            lemma, sl = free[0], ll.max().item()  # compound: keep the lemma of the free prediction
        else:
            r = v["lemma"](ud.lemma_rule(form, lemma))
            sl = ll[r].item() if r else ll.min().item()
        for (u, f) in table[key]:
            iu, i_f = v["upos"](u), v["feats"](f)
            if not iu or not i_f:
                continue
            sc = sl + lu[iu].item() + lf[i_f].item()
            if best is None or (sc, lemma, u, f) > best[0]:
                best = ((sc, lemma, u, f), (lemma, u, f))
    return best[1] if best else None


@torch.no_grad()
def run(path):
    from transformers import AutoTokenizer
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a, v = ck["args"], {k: re.vocab(x) for k, x in ck["vocabs"].items()}
    train_tbs = [a["train"][0]] if a["setting"] == "mono" else [tb for tb in ud.TB if tb in ud.FST_LANG]
    table = tag_table(train_tbs)
    model = ud.Parser(a["encoder"], v, steps=a["steps"])
    model.load_state_dict(ck["state"])
    model.eval().cuda()
    tok = AutoTokenizer.from_pretrained(a["encoder"])
    res = {}
    for tb in re.tests_of(a):
        sents = ud.split(tb, "test")
        items = ud.encode(sents, tok, v)
        c = Counter()
        for i in range(0, len(items), 32):
            batch = [x.cuda() for x in ud.collate(items[i:i + 32], tok.pad_token_id)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pu, pf, pl = [F.log_softmax(x.float(), -1).cpu() for x in model(*batch[:4])[:3]]
            for b, s in enumerate(sents[i:i + 32]):
                for j, t in enumerate(s):
                    c["n"] += 1
                    if j >= len(items[i + b][1]) or items[i + b][1][j] < 0:
                        continue
                    u, f, r = pu[b, j].argmax().item(), pf[b, j].argmax().item(), pl[b, j].argmax().item()
                    lemma = ud.apply_rule(t["form"], v["lemma"].itos[r]) if r else t["form"]
                    lemma = ud.fst_lemma(t["form"], lemma, ud.fst_lemmas(tb, t["form"]), pl[b, j], v)
                    free = (lemma, v["upos"].itos[u], v["feats"].itos[f])
                    joint = decode(t["form"], tb, table, pu[b, j], pf[b, j], pl[b, j], v, free) or free
                    c["joint_used"] += joint is not free
                    gold = (t["lemma"].lower(), t["upos"], t["feats"])
                    for name, p in (("free", free), ("joint", joint)):
                        c[f"{name}_lemma"] += p[0].lower() == gold[0]
                        c[f"{name}_upos"] += p[1] == gold[1]
                        c[f"{name}_feats"] += p[2] == gold[2]
                        c[f"{name}_all"] += (p[0].lower(), p[1], p[2]) == gold
        res[tb] = {k: round(100 * c[k] / c["n"], 2) for k in sorted(c) if k != "n"}
        print(os.path.basename(path), tb, res[tb], flush=True)
    return dict(args=a, table_size=len(table), results=res)


def main():
    names = sys.argv[1:]
    os.makedirs(OUT, exist_ok=True)
    for path in sorted(glob.glob(f"{W}/checkpoints/ud/*.pt")):
        name = os.path.basename(path)[:-3]
        out = f"{OUT}/{name}.json"
        if (names and name not in names) or os.path.exists(out):
            continue
        json.dump(run(path), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
