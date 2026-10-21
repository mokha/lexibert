"""Convert the saved checkpoints into release folders: <name>/{lexibert.safetensors, lexibert.json, vocabs.json,
config.json (encoder), tokenizer files}. The weights are copied unchanged (float32)."""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from lexibert.common import save

W = os.environ.get("LEXIBERT_WORK", "work")
OUT = f"{W}/release/models"
ENC = f"{W}/release/hf/lexibert"  # encoder config + tokenizer of LexiBERT
UD_TBS = ["myv", "mdf", "kpv_lattice", "kpv_ikdp", "koi", "sms", "olo", "krl", "sme", "vep", "yrk"]
LANGS = ["myv", "mdf", "kpv", "koi", "udm", "sme", "sms", "smn", "olo", "krl", "vep", "liv", "mhr", "mrj", "yrk"]


def ck(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    c = ck(f"{W}/models/oov_pt_gloss/model.pt")
    save(f"{OUT}/lexibert-unknown-words", c["state"],
         {"task": "unknown-words", "encoder": "LexiBERT", "languages": LANGS, "input": "'<lang>: <form>'"}, c["vocabs"], ENC)
    for setting, tbs in [("multi", UD_TBS), ("myv_mono", ["myv"]), ("sme_mono", ["sme"])]:
        for seed in (1, 2, 3):
            p = f"{W}/checkpoints/ud/pt_gloss_{setting}_s{seed}.pt"
            if not os.path.exists(p):
                continue
            c = ck(p)
            name = "lexibert-ud-uralic" if setting == "multi" else f"lexibert-ud-{setting.split('_')[0]}"
            save(f"{OUT}/{name}" + (f"-s{seed}" if seed > 1 else ""), c["state"],
                 {"task": "ud-parser", "encoder": "LexiBERT", "trained_on": tbs + (["fin", "est"] if setting == "multi" else []),
                  "seed": seed, "steps": c["args"]["steps"], "deprel": "without subtypes"}, c["vocabs"], ENC)
    for seed in (1, 2, 3):
        c = ck(f"{W}/checkpoints/sentiment/pt_gloss_s{seed}.pt")
        save(f"{OUT}/lexibert-sentiment-sst2" + (f"-s{seed}" if seed > 1 else ""), c["state"],
             {"task": "sentiment", "encoder": "LexiBERT", "trained_on": "SST-2 (English)", "seed": seed,
              "labels": ["negative", "positive"]}, {}, ENC)
        print("done", seed, flush=True)


if __name__ == "__main__":
    main()
