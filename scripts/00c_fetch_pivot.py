"""Stream ~300k sentences of fin / rus / eng web text (FineWeb-2 / FineWeb) for pivot-language teacher vectors."""
import os, re, sys
from datasets import load_dataset

O = os.environ.get("LEXIBERT_WORK", "work") + "/data/corpus_pivot"
os.makedirs(O, exist_ok=True)
SRC = {"fin": ("HuggingFaceFW/fineweb-2", "fin_Latn"), "rus": ("HuggingFaceFW/fineweb-2", "rus_Cyrl"),
       "eng": ("HuggingFaceFW/fineweb", "sample-10BT")}
for lang in sys.argv[1:]:
    repo, cfg = SRC[lang]
    ds = load_dataset(repo, name=cfg, split="train", streaming=True)
    n = 0
    with open(f"{O}/{lang}.tmp", "w", encoding="utf-8") as out:
        for doc in ds:
            for s in re.split(r"(?<=[.!?])\s+", doc["text"].replace("\n", " ")):
                if 4 <= len(s.split()) <= 40:
                    out.write(s.strip() + "\n"); n += 1
            if n >= 300_000:
                break
    os.rename(f"{O}/{lang}.tmp", f"{O}/{lang}.txt")
    print(lang, n, flush=True)
