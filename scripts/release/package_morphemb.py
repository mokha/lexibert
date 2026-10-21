"""Split the MorphEmb student into a shared part (character n-gram and tag vectors) and one lemma table per language,
stored as float16 safetensors. A language needs shared.safetensors + lemmas/<lang>.{safetensors,txt}."""
import json, os, sys
import torch
from safetensors.torch import save_file

W = os.environ.get("LEXIBERT_WORK", "work")


def main(name, out):
    ck = torch.load(f"{W}/emb/{name}/student.pt", map_location="cpu", weights_only=False)
    st = ck["state"]
    os.makedirs(f"{out}/lemmas", exist_ok=True)
    save_file({"char": st["char.weight"].half().contiguous(), "tag": st["tag.weight"].half().contiguous()}, f"{out}/shared.safetensors")
    by_lang = {}
    for key, i in ck["lemma_idx"].items():
        lang, lemma = key.split("|", 1)
        assert "\n" not in lemma
        by_lang.setdefault(lang, []).append((lemma, i))
    for lang, items in sorted(by_lang.items()):
        idx = torch.tensor([i for _, i in items])
        save_file({"lemma": st["lemma.weight"][idx].half().contiguous()}, f"{out}/lemmas/{lang}.safetensors")
        with open(f"{out}/lemmas/{lang}.txt", "w", encoding="utf-8") as f:
            f.writelines(l + "\n" for l, _ in items)
        print(lang, len(items), flush=True)
    json.dump({"model": "MorphEmb", "encoder": "LexiBERT", "dimension": st["char.weight"].shape[1],
               "buckets": st["char.weight"].shape[0], "ngrams": [3, 6], "hash": "zlib.crc32(utf-8) % buckets",
               "max_analyses": 6, "pivot_languages": ["eng", "fin", "rus"], "languages": sorted(by_lang),
               "tags": ck["tags"]}, open(f"{out}/config.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
