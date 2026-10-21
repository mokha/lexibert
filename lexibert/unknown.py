"""Unknown word model: word form -> ranked lemmas, part of speech, tags and inflection class (contlex)."""
import torch, torch.nn as nn
from .common import default_device, encoder_from_config, load, load_weights


def apply_rule(form, rule):
    """Edit rule 'case|k|suffix': remove k characters from the end of the (lowered) form and add the suffix."""
    case, k, s = rule.split("|", 2)
    f = form.lower() if case == "L" else form
    k = int(k)
    return (f[:-k] if k else f) + s if k <= len(f) else None


class UnknownWordModel(nn.Module):
    def __init__(self, folder, vocabs):
        super().__init__()
        self.enc = encoder_from_config(folder)
        H = self.enc.config.hidden_size
        self.heads = nn.ModuleDict({k: nn.Sequential(nn.Dropout(0.1), nn.Linear(H, len(v))) for k, v in vocabs.items()})
        self.vocabs = vocabs

    @classmethod
    def from_pretrained(cls, path, device=None):
        folder, meta, vocabs, tok = load(path)
        device = default_device(device)
        model = load_weights(cls(folder, vocabs), folder, device)
        model.tok, model.meta, model.device = tok, meta, device
        return model

    def forward(self, ids, att):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        pooled = (h * att[..., None]).sum(1) / att.sum(1, keepdim=True)
        return {k: head(pooled) for k, head in self.heads.items()}

    @torch.no_grad()
    def analyse(self, lang, forms, k=5, batch_size=256):
        """Analyse one form or a list of forms of the language `lang` (ISO 639-3, e.g. 'myv').
        Returns, for each form, the k best lemmas, parts of speech and tags and the 3 best inflection classes,
        each as (value, probability). The tags are given without the part of speech."""
        single = isinstance(forms, str)
        forms = [forms] if single else list(forms)
        out = []
        for i in range(0, len(forms), batch_size):
            chunk = forms[i:i + batch_size]
            e = self.tok([f"{lang}: {f}" for f in chunk], padding=True, truncation=True, max_length=32,
                         return_tensors="pt").to(self.device)
            logits = self(e.input_ids, e.attention_mask)  # float32, as in the paper
            top = {h: logits[h].float().softmax(-1).topk(3 if h == "contlex" else k) for h in logits}
            for b, form in enumerate(chunk):
                res = {}
                for h, (p, idx) in top.items():
                    vals = []
                    for pr, j in zip(p[b].tolist(), idx[b].tolist()):
                        if j == 0:  # <unk>
                            continue
                        v = self.vocabs[h][j]
                        if h == "rule":
                            v = apply_rule(form, v)
                            if v is None:
                                continue
                        elif h == "tags":
                            v = v.split("|", 1)[1]
                        elif h == "contlex":
                            if not v.startswith(lang + "|"):
                                continue
                            v = v.split("|", 1)[1]
                        vals.append((v, round(pr, 4)))
                    if h == "rule":  # different rules can give the same lemma (e.g. with and without lower-casing)
                        merged = {}
                        for x, pr in vals:
                            merged[x] = round(merged.get(x, 0) + pr, 4)
                        vals = sorted(merged.items(), key=lambda x: -x[1])
                    res["lemma" if h == "rule" else h] = vals
                out.append(res)
        return out[0] if single else out
