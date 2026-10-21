"""Sentiment classifier trained on English SST-2 only (0 = negative, 1 = positive)."""
import torch, torch.nn as nn
from .common import autocast, default_device, encoder_from_config, load, load_weights


class SentimentClassifier(nn.Module):
    labels = ["negative", "positive"]

    def __init__(self, folder):
        super().__init__()
        self.enc = encoder_from_config(folder)
        self.out = nn.Linear(self.enc.config.hidden_size, 2)

    @classmethod
    def from_pretrained(cls, path, device=None):
        folder, meta, _, tok = load(path)
        device = default_device(device)
        model = load_weights(cls(folder), folder, device)
        model.tok, model.meta, model.device = tok, meta, device
        return model

    def forward(self, ids, att):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        return self.out((h * att[..., None]).sum(1) / att.sum(1, keepdim=True))

    @torch.no_grad()
    def predict(self, texts, batch_size=64):
        out = []
        for i in range(0, len(texts), batch_size):
            e = self.tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            with autocast(self.device):
                out += [self.labels[j] for j in self(e.input_ids, e.attention_mask).argmax(-1).tolist()]
        return out
