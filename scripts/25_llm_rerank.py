"""LLM reranking of the translation candidates of 09_linkpred (run it first with LP_DUMP=work/runs_llm/lp_dump.json).

For every test word, the LLM sees the word, its part of speech, its remaining translations in other languages in the
Ve'rdd graph (the held-out links to the target language are not among them) and the ten best candidates of the ranker.
It answers with the number of the correct candidate, or 0 and its own translation, which is used when it is a word of
the target language in Ve'rdd. The chosen word is moved to the first place; the rest keep the order of the ranker.
Two ways of using the answer are scored: the LLM only chooses among the candidates, or its own word is also used.
Which one to use is decided on the dev words (--split dev; ranked by a ranker trained on the other half of the
ranker's training words), and the test words are then run once.
The model is served by an OpenAI-compatible server (LM Studio / llama.cpp), and every answer is cached.
  python scripts/25_llm_rerank.py [--url http://localhost:1234/v1] [--model NAME] [--split dev|test] [--limit N]"""
import argparse, collections, hashlib, json, os, random, re, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

W = os.environ.get("LEXIBERT_WORK", "work")
LANG = {"myv": "Erzya", "mdf": "Moksha", "kpv": "Komi-Zyrian", "koi": "Komi-Permyak", "udm": "Udmurt",
        "sme": "North Sámi", "sms": "Skolt Sámi", "smn": "Inari Sámi", "sma": "South Sámi", "smj": "Lule Sámi",
        "olo": "Livvi", "krl": "Karelian", "liv": "Livonian", "mhr": "Meadow Mari", "mrj": "Hill Mari",
        "vep": "Veps", "vot": "Votic", "izh": "Ingrian", "fin": "Finnish", "est": "Estonian", "hun": "Hungarian",
        "rus": "Russian", "eng": "English", "deu": "German", "fra": "French", "nob": "Norwegian", "swe": "Swedish",
        "yrk": "Nenets", "kca": "Khanty", "mns": "Mansi", "lav": "Latvian", "spa": "Spanish"}
POS = {"N": "noun", "A": "adjective", "V": "verb", "Adv": "adverb"}
SYSTEM = ("You are an expert lexicographer of the Uralic languages. You are given a word, its translations into other "
          "languages in a dictionary and a numbered list of candidate translations into a target language. Answer with "
          "the number of the correct translation. If none of them is correct, answer 0 and the correct translation, as "
          "in \"0: word\". Answer briefly, without explanations.")


def prompt(d):
    L, T = d["pair"].split("-")
    known = collections.OrderedDict()
    for lg, w in d["known"]:
        known.setdefault(LANG.get(lg, lg), []).append(w)
    lines = [f"{LANG[L]} word: {d['src']}" + (f" ({POS[d['pos']]})" if d["pos"] in POS else "")]
    if known:
        lines.append("Its translations into other languages in the dictionary: " +
                     "; ".join(f"{k}: {', '.join(v[:4])}" for k, v in list(known.items())[:12]))
    lines.append(f"Which of these {LANG[T]} words is its translation?")
    lines += [f"{k + 1}. {w}" for k, (_, w) in enumerate(d["top"])]
    return "\n".join(lines)


class Client:
    def __init__(self, url, model, cache):
        self.url, self.model, self.path, self.lock = url, model, cache, threading.Lock()
        self.cache = {}
        if os.path.exists(cache):
            for line in open(cache, encoding="utf-8"):
                r = json.loads(line); self.cache[r["key"]] = r["answer"]

    def __call__(self, text):
        key = hashlib.sha256(f"{self.model}\n{SYSTEM}\n{text}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        body = dict(model=self.model, temperature=0, max_tokens=400, seed=0,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}])
        req = urllib.request.Request(f"{self.url}/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
        answer = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        with self.lock:
            self.cache[key] = answer
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(key=key, prompt=text, answer=answer), ensure_ascii=False) + "\n")
        return answer


def parse(answer, n):
    m = re.match(r"\s*\**\s*(\d+)\s*\**\s*(?:[:.\-–)]\s*(.+))?", answer)
    if not m:
        return None, None
    k = int(m.group(1))
    free = (m.group(2) or "").strip().strip("\"'*.").strip() if k == 0 else None
    return (k if 1 <= k <= n else None), (free or None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:1234/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--limit", type=int, default=0, help="random words per pair (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--dump", default="lp_dump.json", help="ranker dump under work/runs_llm/")
    ap.add_argument("--tag", default="", help="suffix of the result file")
    ap.add_argument("--cached-only", action="store_true", help="score only the words already answered (a random sample, "
                    "as the words are asked in random order)")
    a = ap.parse_args()
    dump = [d for d in json.load(open(f"{W}/runs_llm/{a.dump}")) if d["split"] == a.split]
    lx = pd.read_parquet(f"{W}/data/verdd/lexemes.parquet", columns=["id", "lang", "lexeme"])
    by_text = {(l, w.lower()): int(i) for i, l, w in zip(lx.id, lx.lang, lx.lexeme)}
    pairs = collections.defaultdict(list)
    for d in dump:
        pairs[d["pair"]].append(d)
    if a.limit:
        pairs = {p: random.Random(0).sample(ds, min(a.limit, len(ds))) for p, ds in pairs.items()}
    llm = Client(a.url, a.model, f"{W}/runs_llm/cache_{a.model.replace('/', '_')}.jsonl")
    todo = [d for ds in pairs.values() for d in ds if d["top"]]
    random.Random(0).shuffle(todo)  # random order, so that a partial run is a random sample
    if a.cached_only:
        key = lambda d: hashlib.sha256(f"{a.model}\n{SYSTEM}\n{prompt(d)}".encode()).hexdigest()
        done = {id(d) for d in todo if key(d) in llm.cache}
        pairs = {p: [d for d in ds if id(d) in done] for p, ds in pairs.items()}
        pairs = {p: ds for p, ds in pairs.items() if ds}
        todo = [d for d in todo if id(d) in done]
    with ThreadPoolExecutor(a.workers) as ex:
        answers = dict(zip((id(d) for d in todo), ex.map(lambda d: llm(prompt(d)), todo)))
    out = {}
    for p, ds in pairs.items():
        T = p.split("-")[1]
        n = len(ds) if a.limit or a.cached_only else ds[0]["n_test_src"]  # words without candidates count as misses
        c = collections.Counter()
        for d in ds:
            gold = set(d["gold_ids"])
            top = [cid for cid, _ in d["top"]]
            c["ranker"] += bool(top) and top[0] in gold
            c["ranker@10"] += bool(set(top) & gold)
            pick = free_first = top[0] if top else None
            if top:
                k, free = parse(answers[id(d)], len(top))
                c["parsed"] += k is not None or free is not None
                if k:
                    pick = free_first = top[k - 1]
                elif free and (T, free.lower()) in by_text:
                    free_first = by_text[(T, free.lower())]; c["free_used"] += 1
            c["llm_pick"] += pick in gold  # the LLM chooses among the candidates only
            c["llm_free"] += free_first in gold  # ... or gives its own word of the target language
        out[p] = {"n": n, "ranker_hits@1": round(100 * c["ranker"] / n, 1), "ranker_hits@10": round(100 * c["ranker@10"] / n, 1),
                  "llm_pick_hits@1": round(100 * c["llm_pick"] / n, 1), "llm_free_hits@1": round(100 * c["llm_free"] / n, 1),
                  "parsed": c["parsed"], "free_used": c["free_used"]}
        print(p, out[p], flush=True)
    avg = lambda k: round(sum(v[k] for v in out.values()) / len(out), 1)
    out["average"] = {k: avg(k) for k in ("ranker_hits@1", "ranker_hits@10", "llm_pick_hits@1", "llm_free_hits@1")}
    print("average", out["average"])
    json.dump(dict(model=a.model, split=a.split, limit=a.limit, results=out),
              open(f"{W}/runs_llm/rerank_{a.model.replace('/', '_')}_{a.split}{a.tag}{'_lim' + str(a.limit) if a.limit else ''}{'_partial' if a.cached_only else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
