"""Export the MorphEmb vectors of every language as gzipped fastText .vec files (4 decimals).
The words are the same as in 11_morph_embeddings.export: the corpus types with a teacher vector and the types that the
FST (or the unknown word model) can analyse."""
import gzip, os, sys
from importlib import import_module
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
me = import_module("11_morph_embeddings")


def main(name, out):
    src = f"{me.W}/emb/{name}"
    os.makedirs(out, exist_ok=True)
    model, feat = me.load_student(src, me.URALIC)
    for lang in me.URALIC + me.PIVOT:
        path = f"{out}/{lang}.vec.gz"
        if os.path.exists(path):
            continue
        words = []
        p = f"{src}/teacher_{lang}.pt"
        if os.path.exists(p):
            words += torch.load(p, map_location="cpu")["words"]
        if lang in feat.an.a:
            words += list(feat.an.a[lang].keys())[:me.MAX_TYPES]
        words = [w for w in dict.fromkeys(words) if w and not any(c.isspace() for c in w)]
        vecs = me.embed(model, feat, [(lang, w) for w in words], "cuda").numpy()
        with gzip.open(path + ".tmp", "wt", encoding="utf-8", compresslevel=6) as f:
            f.write(f"{len(words)} {me.DIM}\n")
            for w, v in zip(words, vecs):
                f.write(w + " " + " ".join(f"{x:.4f}" for x in v) + "\n")
        os.replace(path + ".tmp", path)
        print("export", lang, len(words), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
