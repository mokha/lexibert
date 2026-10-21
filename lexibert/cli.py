"""Command line tools.
  lexibert-parse --model DIR_OR_HF_ID [--fst LANG] < input > output.conllu
  lexibert-unknown --model DIR_OR_HF_ID --lang myv WORD [WORD ...]   (or words from stdin, one per line)"""
import argparse, json, sys


def parse_main():
    from .parser import Parser, read_input, to_conllu
    ap = argparse.ArgumentParser(description="Parse CoNLL-U or tokenized text (one sentence per line).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--fst", metavar="LANG", help="restrict the lemmas to those of the FST of LANG (needs uralicNLP)")
    ap.add_argument("--device")
    ap.add_argument("input", nargs="?", type=argparse.FileType(encoding="utf-8"), default=sys.stdin)
    a = ap.parse_args()
    p = Parser.from_pretrained(a.model, a.device)
    sys.stdout.write(to_conllu(p.parse(read_input(a.input), fst_lang=a.fst)))


def unknown_main():
    from .unknown import UnknownWordModel
    ap = argparse.ArgumentParser(description="Guess the lemma, tags and inflection class of word forms.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device")
    ap.add_argument("words", nargs="*")
    a = ap.parse_args()
    words = a.words or [l.strip() for l in sys.stdin if l.strip()]
    m = UnknownWordModel.from_pretrained(a.model, a.device)
    for w, r in zip(words, m.analyse(a.lang, words, k=a.k)):
        print(json.dumps({"form": w, **r}, ensure_ascii=False))
