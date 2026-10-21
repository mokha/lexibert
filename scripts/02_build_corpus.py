"""Build per-language plain-text corpora from our raw text (+ FineWeb2 if present),
deduplicated and decontaminated against every UD dev/test sentence."""
import glob, os, re, sys, unicodedata, html
from collections import defaultdict

W = os.environ.get("LEXIBERT_WORK", "work")
SRC = f"{W}/data/raw_text"
OUT = f"{W}/data/corpus"
FW2 = f"{W}/data/fineweb2/data"
UD_LANG = {"Erzya": "myv", "Moksha": "mdf", "Komi_Zyrian": "kpv", "Komi_Permyak": "koi",
           "Skolt_Sami": "sms", "Livvi": "olo", "Karelian": "krl", "North_Sami": "sme",
           "Veps": "vep", "Nenets": "yrk", "Finnish": "fin", "Estonian": "est"}
FW2_LANG = {"myv_Cyrl": "myv", "mdf_Cyrl": "mdf", "kpv_Cyrl": "kpv", "koi_Cyrl": "koi", "udm_Cyrl": "udm",
            "sme_Latn": "sme", "sms_Latn": "sms", "smn_Latn": "smn", "olo_Latn": "olo", "krl_Latn": "krl",
            "vep_Latn": "vep", "liv_Latn": "liv", "mhr_Cyrl": "mhr", "mrj_Cyrl": "mrj", "yrk_Cyrl": "yrk",
            "mns_Cyrl": "mns"}
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[\"«„(]?[A-ZА-ЯЁÄÖÕÜŠŽČĐŊÁ0-9])")


def norm(s):
    s = unicodedata.normalize("NFC", html.unescape(s)).replace("﻿", "")
    return re.sub(r"\s+", " ", s).strip()


def key(s):
    """Aggressive normalisation used only for contamination matching."""
    return re.sub(r"[\W_]+", "", s.lower())


def ud_keys():
    keys = set()
    for f in glob.glob(f"{W}/data/ud/UD_*/*-ud-*.conllu"):
        if "-ud-train" in f:
            continue
        for line in open(f, encoding="utf-8"):
            if line.startswith("# text ="):
                k = key(line.split("=", 1)[1])
                if len(k) >= 12:
                    keys.add(k)
    return keys


def from_token_xml(text):
    for sent in re.findall(r"<sentence[^>]*>(.*?)</sentence>", text, re.S):
        yield " ".join(html.unescape(t) for t in re.findall(r"<token[^>]*>(.*?)</token>", sent))


def from_text(text):
    text = re.sub(r"<[^>]+>", "\n", text)
    for line in text.split("\n"):
        line = norm(line)
        if not line or line.startswith("{") or re.fullmatch(r"[=\W\d]*", line):
            continue
        line = line.strip("= ").strip()
        yield from SENT_SPLIT.split(line)


def raw_text(lang):
    for f in sorted(glob.glob(f"{SRC}/{lang}/**/*", recursive=True)):
        if not os.path.isfile(f):
            continue
        text = open(f, encoding="utf-8", errors="replace").read()
        yield from (from_token_xml(text) if "<token" in text[:5000] else from_text(text))


def fineweb2(lang):
    import pyarrow.parquet as pq
    for code, l in FW2_LANG.items():
        if l != lang:
            continue
        for f in sorted(glob.glob(f"{FW2}/{code}/**/*.parquet", recursive=True)):
            if "_removed" in f:
                continue
            for doc in pq.read_table(f, columns=["text"]).column("text").to_pylist():
                yield from from_text(doc)


def main(langs):
    os.makedirs(OUT, exist_ok=True)
    contaminated = ud_keys()
    print("UD dev/test sentence keys:", len(contaminated))
    for lang in langs:
        seen, stats = set(), defaultdict(int)
        with open(f"{OUT}/{lang}.txt", "w", encoding="utf-8") as out:
            for source, gen in (("raw_text", raw_text(lang)), ("fineweb2", fineweb2(lang))):
                for s in gen:
                    s = norm(s)
                    k = key(s)
                    if len(k) < 3 or len(s.split()) < 2:
                        continue
                    if k in contaminated:
                        stats["removed_ud"] += 1
                        continue
                    if k in seen:
                        continue
                    seen.add(k)
                    stats[source] += 1
                    stats["tokens"] += len(s.split())
                    out.write(s + "\n")
        print(lang, dict(stats), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["myv", "mdf", "kpv", "koi", "udm", "sme", "sms", "smn", "olo", "krl",
                          "vep", "liv", "mhr", "mrj", "yrk", "mns"])
