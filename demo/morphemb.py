"""MorphEmb vectors (the same computation as `load_morphemb` in MLTools): the average of the character n-gram vectors
of a word plus the average over its FST analyses of the lemma vector and the average tag vector."""
import json, zlib
from pathlib import Path
import torch, torch.nn.functional as F
from safetensors.torch import load_file

PIVOTS = {"eng", "fin", "rus"}
LANG_NAME = {}  # filled in by the app, for the progress messages


def ngrams(word, low, high):
    marked = f"<{word}>"
    return [marked[i:i + n] for n in range(low, high + 1) for i in range(len(marked) - n + 1)]


class MorphEmb:
    def __init__(self, folder, analyze):
        self.folder = Path(folder)
        self.config = json.loads((self.folder / "config.json").read_text(encoding="utf-8"))
        shared = load_file(str(self.folder / "shared.safetensors"))
        self.char, self.tag = shared["char"], shared["tag"]
        self.tag_ids = {t: i for i, t in enumerate(self.config["tags"])}
        self.analyze = analyze  # (word, lang) -> ["lemma+TAG+TAG", ...]
        self.lemma_vectors, self.lemma_ids, self.lemma_lists, self.matrices = {}, {}, {}, {}

    def load(self, lang):
        if lang not in self.lemma_ids:
            self.lemma_vectors[lang] = load_file(str(self.folder / "lemmas" / f"{lang}.safetensors"))["lemma"]
            lemmas = (self.folder / "lemmas" / f"{lang}.txt").read_text(encoding="utf-8").split("\n")[:-1]
            self.lemma_lists[lang], self.lemma_ids[lang] = lemmas, {l: i for i, l in enumerate(lemmas)}

    def analyses(self, word, lang):
        if lang in PIVOTS:
            return []
        return list(self.analyze(word, lang)) or (list(self.analyze(word.lower(), lang)) if word != word.lower() else [])

    def vector(self, word, lang, analyses=None):
        self.load(lang)
        low, high = self.config["ngrams"]
        ids = [zlib.crc32(g.encode()) % self.config["buckets"] for g in ngrams(word, low, high)] or [0]
        vec = self.char[torch.tensor(ids)].float().mean(0)
        parts, none = [], self.tag_ids["<none>"]
        if lang in PIVOTS:
            i = self.lemma_ids[lang].get(word.lower())
            if i is not None:
                parts.append(self.lemma_vectors[lang][i].float() + self.tag[none].float())
        for a in (self.analyses(word, lang) if analyses is None else analyses)[:self.config["max_analyses"]]:
            pieces = [p for p in a.split("#")[-1].split("+") if p]
            i = self.lemma_ids[lang].get(pieces[0]) if pieces else None
            if i is None:
                continue
            tags = [self.tag_ids.get(t, none) for t in pieces[1:]] or [none]
            parts.append(self.lemma_vectors[lang][i].float() + self.tag[torch.tensor(tags)].float().mean(0))
        if parts:
            vec = vec + torch.stack(parts).mean(0)
        return vec

    def matrix(self, lang, progress=None):
        """Normalised vectors of all lemmas of `lang`, computed once and in batches (the same sums as `vector`)."""
        if lang in self.matrices:
            return self.matrices[lang]
        self.load(lang)
        words, ids, none = self.lemma_lists[lang], self.lemma_ids[lang], self.tag_ids["<none>"]
        low, high = self.config["ngrams"]
        out = []
        for start in range(0, len(words), 4096):
            chunk = words[start:start + 4096]
            if progress is not None:
                progress(start / len(words), desc=f"Building the {LANG_NAME.get(lang, lang)} lemma vectors (only once)")
            grams, offsets = [], []
            parts_word, parts_lemma, tag_ids, tag_offsets = [], [], [], []
            for k, w in enumerate(chunk):
                offsets.append(len(grams))
                grams += [zlib.crc32(g.encode()) % self.config["buckets"] for g in ngrams(w, low, high)] or [0]
                if lang in PIVOTS:
                    i = ids.get(w.lower())
                    analyses = [] if i is None else [(i, [none])]
                else:
                    analyses = []
                    for a in self.analyses(w, lang)[:self.config["max_analyses"]]:
                        pieces = [p for p in a.split("#")[-1].split("+") if p]
                        i = ids.get(pieces[0]) if pieces else None
                        if i is not None:
                            analyses.append((i, [self.tag_ids.get(t, none) for t in pieces[1:]] or [none]))
                for i, tags in analyses:
                    parts_word.append(k); parts_lemma.append(i)
                    tag_offsets.append(len(tag_ids)); tag_ids += tags
            vec = F.embedding_bag(torch.tensor(grams), self.char.float(), torch.tensor(offsets), mode="mean")
            if parts_word:
                part = self.lemma_vectors[lang][torch.tensor(parts_lemma)].float() + F.embedding_bag(
                    torch.tensor(tag_ids), self.tag.float(), torch.tensor(tag_offsets), mode="mean")
                pw = torch.tensor(parts_word)
                total = torch.zeros_like(vec).index_add_(0, pw, part)
                count = torch.zeros(len(chunk)).index_add_(0, pw, torch.ones(len(pw)))
                vec = vec + total / count.clamp(min=1)[:, None]
            out.append(F.normalize(vec, dim=1).half())
        self.matrices[lang] = torch.cat(out)
        return self.matrices[lang]

    def neighbours(self, word, lang, target, n=10, progress=None, limit=None):
        """The n nearest lemmas of `target`; with `limit`, only among its `limit` first lemmas (the lemma lists are
        roughly in order of frequency)."""
        m = self.matrix(target, progress)[:limit]
        scores = m.float() @ F.normalize(self.vector(word, lang), dim=0)
        vals, idx = torch.topk(scores, min(n + 1, len(scores)))
        words = self.lemma_lists[target]
        return [(words[i], round(float(v), 3)) for v, i in zip(vals, idx.tolist())
                if not (target == lang and words[i] == word)][:n]

    def similarity(self, a, lang_a, b, lang_b):
        return float(F.cosine_similarity(self.vector(a, lang_a), self.vector(b, lang_b), dim=0))
