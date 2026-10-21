"""Baseline: fastText (gensim) trained on exactly the same decontaminated corpora as morph_embeddings."""
import os, re, sys
from gensim.models import FastText

W = os.environ.get("LEXIBERT_WORK", "work")
CORE = re.compile(r"[^\W\d_]+(?:[’'ʼʹ-][^\W\d_]+)*")


class Sents:
    def __init__(self, path, cap=3_000_000):
        self.path, self.cap = path, cap

    def __iter__(self):
        for i, line in enumerate(open(self.path, encoding="utf-8")):
            if i >= self.cap:
                break
            yield CORE.findall(line)


os.makedirs(f"{W}/emb/fasttext", exist_ok=True)
for lang in sys.argv[1:]:
    m = FastText(vector_size=300, window=5, min_count=2, sg=1, epochs=5, workers=14, min_n=3, max_n=6, bucket=2_000_000)
    s = Sents(f"{W}/data/corpus/{lang}.txt")
    m.build_vocab(corpus_iterable=s)
    m.train(corpus_iterable=s, total_examples=m.corpus_count, epochs=m.epochs)
    m.save(f"{W}/emb/fasttext/{lang}.model")
    m.wv.save_word2vec_format(f"{W}/emb/fasttext/{lang}.vec")
    print(lang, len(m.wv), flush=True)
