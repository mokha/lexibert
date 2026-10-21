import json, os, shutil
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

WEIGHTS, META, VOCABS = "lexibert.safetensors", "lexibert.json", "vocabs.json"


def resolve(path):
    """A local folder or a Hugging Face model id."""
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download
    return snapshot_download(path)


def encoder_from_config(folder):
    return AutoModel.from_config(AutoConfig.from_pretrained(folder), attn_implementation="sdpa")


def load(folder):
    folder = resolve(folder)
    meta = json.load(open(os.path.join(folder, META), encoding="utf-8"))
    vocabs = json.load(open(os.path.join(folder, VOCABS), encoding="utf-8")) if os.path.exists(os.path.join(folder, VOCABS)) else {}
    return folder, meta, vocabs, AutoTokenizer.from_pretrained(folder)


def load_weights(module, folder, device):
    module.load_state_dict(load_file(os.path.join(folder, WEIGHTS)))
    return module.to(device).eval()


def default_device(device=None):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def autocast(device):
    """bfloat16 on the GPU, as in the experiments of the paper."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)


def save(folder, state, meta, vocabs, encoder_dir):
    """Used by scripts/release/package_models.py: weights + metadata + the encoder config and tokenizer."""
    os.makedirs(folder, exist_ok=True)
    save_file({k: v.detach().contiguous().clone() for k, v in state.items()}, os.path.join(folder, WEIGHTS))
    json.dump(meta, open(os.path.join(folder, META), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if vocabs:
        json.dump(vocabs, open(os.path.join(folder, VOCABS), "w", encoding="utf-8"), ensure_ascii=False)
    cfg = json.load(open(os.path.join(encoder_dir, "config.json"), encoding="utf-8"))
    cfg["architectures"] = ["ModernBertModel"]
    json.dump(cfg, open(os.path.join(folder, "config.json"), "w", encoding="utf-8"), indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):  # as released with mmBERT-small
        shutil.copyfile(os.path.join(encoder_dir, f), os.path.join(folder, f))


def fst_analyses(lang):
    """word -> FST analyses with UralicNLP (the descriptive analysers, as in the paper). Capitalised words that the
    FST does not know are also tried in lower case."""
    from uralicNLP import uralicApi

    def analyse(word):
        res = uralicApi.analyze(word, lang)
        if not res and word[:1].isupper():
            res = uralicApi.analyze(word.lower(), lang)
        return [a for a, _ in res]
    return analyse
