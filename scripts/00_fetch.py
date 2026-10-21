"""Fetch every external resource the experiments need into work/."""
import os, sys, traceback
W = os.environ.get("LEXIBERT_WORK", "work")
os.environ["HF_HOME"] = f"{W}/hf"
LANGS = ["myv", "mdf", "kpv", "koi", "udm", "sme", "sms", "smn", "olo", "krl", "vep",
         "liv", "mhr", "mrj", "yrk", "mns", "fin", "est"]

def step(name, fn):
    print(f"== {name}", flush=True)
    try: fn()
    except Exception: traceback.print_exc()

def fsts():
    from uralicNLP import uralicApi
    for l in LANGS:
        try:
            uralicApi.download(l, show_progress=False); print("fst ok", l, flush=True)
        except Exception as e: print("fst FAIL", l, e, flush=True)

def models():
    from huggingface_hub import snapshot_download
    for r in ["jhu-clsp/mmBERT-small", "jhu-clsp/mmBERT-base", "cis-lmu/glot500-base", "FacebookAI/xlm-roberta-base"]:
        snapshot_download(r, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]); print("model ok", r, flush=True)

def wikiann():
    from datasets import load_dataset
    for l in ["en", "fi", "et", "myv", "mdf", "kv", "koi", "udm", "se", "olo", "vep", "mhr", "mrj"]:
        try:
            d = load_dataset("unimelb-nlp/wikiann", l); d.save_to_disk(f"{W}/data/wikiann/{l}")
            print("wikiann", l, {k: len(v) for k, v in d.items()}, flush=True)
        except Exception as e: print("wikiann FAIL", l, e, flush=True)

def fineweb2():
    from huggingface_hub import snapshot_download
    fw = ["myv_Cyrl", "mdf_Cyrl", "kpv_Cyrl", "koi_Cyrl", "udm_Cyrl", "sme_Latn", "sms_Latn", "smn_Latn",
          "olo_Latn", "krl_Latn", "vep_Latn", "liv_Latn", "mhr_Cyrl", "mrj_Cyrl", "yrk_Cyrl", "mns_Cyrl"]
    snapshot_download("HuggingFaceFW/fineweb-2", repo_type="dataset",
                      allow_patterns=[f"data/{l}/*" for l in fw], local_dir=f"{W}/data/fineweb2")
    print("fineweb2 ok", flush=True)

def stanza_models():
    import stanza
    for l in ["myv", "sme", "kpv", "fi", "et", "olo", "krl", "koi", "mdf", "sms"]:
        try: stanza.download(l, model_dir=f"{W}/stanza", logging_level="WARN"); print("stanza ok", l, flush=True)
        except Exception as e: print("stanza FAIL", l, e, flush=True)

for n, f in [("models", models), ("wikiann", wikiann), ("stanza", stanza_models), ("fineweb2", fineweb2)]:
    step(n, f)
print("ALL DONE")
