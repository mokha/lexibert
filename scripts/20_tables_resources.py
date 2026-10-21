"""Generate paper/tables/resources.tex from the data actually used."""
import glob, os, re
import numpy as np, pandas as pd

W = os.environ.get("LEXIBERT_WORK", "work")
OUT = os.environ.get("LEXIBERT_TABLES", "tables")
LANGS = [("myv", "Erzya"), ("mdf", "Moksha"), ("kpv", "Komi-Zyrian"), ("koi", "Komi-Permyak"), ("udm", "Udmurt"),
         ("mhr", "Meadow Mari"), ("mrj", "Hill Mari"), ("sme", "North Sámi"), ("smn", "Inari Sámi"), ("sms", "Skolt Sámi"),
         ("olo", "Livvi"), ("krl", "Karelian"), ("vep", "Veps"), ("liv", "Livonian"), ("yrk", "Tundra Nenets")]
UD = {"myv": ["Erzya-JR"], "mdf": ["Moksha-JR"], "kpv": ["Komi_Zyrian-Lattice", "Komi_Zyrian-IKDP"], "koi": ["Komi_Permyak-UH"],
      "sms": ["Skolt_Sami-Giellagas"], "olo": ["Livvi-KKPP"], "krl": ["Karelian-KKPP"], "sme": ["North_Sami-Giella"],
      "vep": ["Veps-VWT"], "yrk": ["Nenets-Tundra"]}
PIVOTS = {"eng", "fin", "rus"}


def n_sents(path):
    return sum(1 for line in open(path, encoding="utf-8") if line.strip()) if os.path.exists(path) else 0


def ud_sizes(lang):
    tr = te = 0
    for tb in UD.get(lang, []):
        for f in glob.glob(f"{W}/data/ud/UD_{tb}/*-ud-*.conllu"):
            n = sum(1 for l in open(f, encoding="utf-8") if l.startswith("# sent_id"))
            if "-ud-train" in f:
                tr += n
            elif "-ud-test" in f:
                te += n
    return tr, te


def main():
    os.makedirs(OUT, exist_ok=True)
    lx = pd.read_parquet(f"{W}/data/verdd/lexemes.parquet", columns=["id", "lang", "contlex"])
    rel = pd.read_parquet(f"{W}/data/verdd/relations.parquet", columns=["type", "f", "t"])
    rel = rel[rel.type == 0]
    lang = lx.set_index("id").lang
    a, b = rel.f.map(lang), rel.t.map(lang)
    rows = []
    for code, name in LANGS:
        tokens = 0
        p = f"{W}/data/corpus/{code}.txt"
        if os.path.exists(p):
            tokens = sum(len(l.split()) for l in open(p, encoding="utf-8"))
        fst = pd.read_parquet(f"{W}/data/fst/{code}.parquet", columns=["analyses", "freq"])
        cov = fst.freq[fst.analyses.map(len) > 0].sum() / max(1, fst.freq.sum())
        l = lx[lx.lang == code]
        n_cl = (l.contlex.fillna("").str.len() > 2).sum()
        n_tr = int((((a == code) & b.isin(PIVOTS)) | ((b == code) & a.isin(PIVOTS))).sum())
        tr, te = ud_sizes(code)
        rows.append((name, code, tokens / 1e6, 100 * cov, len(l) / 1e3, n_cl / 1e3, n_tr / 1e3, tr, te))
    with open(f"{OUT}/resources.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n")
        f.write("\\begin{adjustbox}{max width=\\linewidth}\\begin{tabular}{llrrrrrrr}\n\\toprule\n")
        f.write(" & & \\multicolumn{2}{c}{Corpus} & \\multicolumn{3}{c}{Ve'rdd (k)} & \\multicolumn{2}{c}{UD sent.}\\\\\n")
        f.write("\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\n")
        f.write("Language & ISO & Mtok & FST\\% & lex. & contlex & trans. & train & test\\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r[0]} & {r[1]} & {r[2]:.2f} & {r[3]:.1f} & {r[4]:.1f} & {r[5]:.1f} & {r[6]:.1f} & "
                    f"{r[7] if r[7] else '--'} & {r[8] if r[8] else '--'}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\\end{adjustbox}\n")
        f.write("\\caption{Resources per language: corpus size after deduplication and UD decontamination (million "
                "whitespace tokens), FST token coverage of that corpus, Ve'rdd lexemes, lexemes with an inflection "
                "class (contlex), translation links to English, Finnish or Russian, and UD sentences.}\n"
                "\\label{tab:resources}\n\\end{table}\n")
    print(open(f"{OUT}/resources.tex").read())


if __name__ == "__main__":
    main()
