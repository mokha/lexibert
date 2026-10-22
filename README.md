# LexiBERT

Code and models of the paper *Grounding Encoders and Word Embeddings in FSTs and Dictionaries for Uralic Languages*
(Alnajjar, Hämäläinen, Rueter and Partanen, IWCLUL 2026).

We continue the pretraining of mmBERT-small on 15 Uralic languages with the help of their FSTs and the Ve'rdd
dictionaries. This repository has the inference code for the released models and the scripts of all experiments.
The data are not distributed.

## Models

| Model | What it does | Size |
|---|---|---|
| [LexiBERT](https://huggingface.co/khalidmokha/lexibert) | The encoder: a standard Hugging Face ModernBERT model | 564 MB |
| [LexiBERT UD](https://huggingface.co/khalidmokha/lexibert-ud-uralic) | UPOS, UFeats, lemmas and dependencies for 11 Uralic treebanks | 576 MB |
| [LexiBERT unknown words](https://huggingface.co/khalidmokha/lexibert-unknown-words) | Lemma, tags and inflection class (contlex) of words that the FST does not know | 613 MB |
| [LexiBERT sentiment](https://huggingface.co/khalidmokha/lexibert-sentiment-sst2) | Sentiment classifier trained on English SST-2 only | 562 MB |
| [MorphEmb](https://huggingface.co/khalidmokha/morphemb) | Static word vectors for 15 Uralic languages, English, Finnish and Russian in one space | 384 MB + lemma tables and vectors per language |

Everything, including the parsers trained on Erzya and North Sámi only, all random seeds, the BiLSTM parsers, the
translation rankers and the encoders of the ablation, is also in the Zenodo record <https://doi.org/10.5281/zenodo.23545487>.

The encoder needs no code from this repository:

```python
from transformers import pipeline
fill = pipeline("fill-mask", model="khalidmokha/lexibert")
fill("Мон вечкан <mask> .")
```

## Demo

`demo/` has a web app for trying all of the models in the browser. In short (the full instructions, also for
Windows, are in [demo/README.md](demo/README.md)):

```bash
git clone https://github.com/mokha/lexibert.git && cd lexibert
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux, CPU only
pip install -e . -r demo/requirements.txt
python demo/app.py                                                   # then open http://127.0.0.1:7860
```

## Installation

```bash
pip install -e .            # the lexibert package: unknown word model, UD parser, sentiment classifier
pip install uralicNLP       # only for restricting the lemmas to the FST
python -c "from uralicNLP import uralicApi; uralicApi.download('myv')"
```

## UD parsing

The input is tokenized: CoNLL-U (the FORM column is used) or one sentence per line with the words separated by
spaces. The output is CoNLL-U with the lemma, UPOS, UFeats, head and relation (without subtypes).

```bash
lexibert-parse --model khalidmokha/lexibert-ud-uralic --fst myv < input.conllu > output.conllu
```

```python
from lexibert import Parser
from lexibert.parser import to_conllu
parser = Parser.from_pretrained("khalidmokha/lexibert-ud-uralic")
print(to_conllu(parser.parse([["Мон", "вечкан", "кудом", "."]], fst_lang="myv")))
```

With `--fst` / `fst_lang`, the lemma must be one of the lemmas that the UralicNLP FST gives for the word, when the FST
knows it (Lem_FST in the paper).

## Unknown words

```bash
lexibert-unknown --model khalidmokha/lexibert-unknown-words --lang myv кудосо
```

```python
from lexibert import UnknownWordModel
model = UnknownWordModel.from_pretrained("khalidmokha/lexibert-unknown-words")
model.analyse("myv", "кудосо")
# {'lemma': [('кудо', 0.9855), ...], 'pos': [('N', 0.9941), ...], 'tags': [('SP+Ine+Indef', 0.9892), ...],
#  'contlex': [('N_KUDO', 0.9406), ...]}
```

## Word embeddings

The `.vec.gz` files have vectors for the most frequent words of our corpora and the words that the FST can analyse,
in fastText text format (e.g. for Gensim). For any other word form, the vector can be built from its FST analyses
with the `load_morphemb` function, which is being added to [MLTools](https://github.com/mikahama/MLTools) and is not
released there yet:

```python
from MLTools.LM import word_embeddings
model = word_embeddings.load_morphemb("morphemb", languages=["myv", "fin"])
model.nearest_neighbors("кудо", "myv", n=5, target_lang="fin")
```

## Sentiment

```python
from lexibert import SentimentClassifier
clf = SentimentClassifier.from_pretrained("khalidmokha/lexibert-sentiment-sst2")
clf.predict(["..."])
```

The paper also glosses the test sentences into English word by word with Ve'rdd before classifying them
(`scripts/26_sentiment_gloss.py`), which is not part of this package.

## Experiments

`run.sh` lists the scripts in the order in which they were run, with the settings of the paper. All scripts read and
write under `$LEXIBERT_WORK` (default `work/`). The versions of the libraries are in `requirements-frozen.txt`
(Python 3.12, one NVIDIA RTX 3090).

The data have to be prepared by yourself, under `$LEXIBERT_WORK`:

- `data/raw_text/<lang>/`: the raw text files read by `scripts/02_build_corpus.py`,
- `data/ud/UD_*`: the UD treebanks,
- `data/sst2/`: SST-2 (`stanfordnlp/sst2`, saved with `datasets.save_to_disk`),
- `downloads/zenodo_7866456/`: the sentiment test data and vectors of Alnajjar et al. (2023)
  ([Zenodo](https://doi.org/10.5281/zenodo.7866456)),
- a MariaDB copy of the Ve'rdd database for `scripts/01_export_verdd.py` (`VERDD_USER`, `VERDD_PASSWORD`,
  `VERDD_HOST`, `VERDD_PORT`).

FineWeb2, mmBERT-small, the Stanza models and the FSTs are downloaded by `scripts/00_fetch.py` and
`scripts/00b_fetch_fsts.py`, and the English, Finnish and Russian web text by `scripts/00c_fetch_pivot.py`.

`scripts/release/` converts the checkpoints into the released folders, exports the vectors and checks that the
released models give the results of the paper.

### Known issues

These are kept as they are so that the scripts reproduce the paper:

- `05_prep_pretrain.py` splits compounds only on `#` when it reads the FST tags. The Karelian FST, and partly the
  Komi-Zyrian one, mark compounds differently, so some word parts end up in the tag vocabulary.
- The BiLSTM parsers of `13_ud_static.py` trained on the CPU vary by about ±0.4 LAS between runs with the same seed.

## Citation

```bibtex
@inproceedings{alnajjar-etal-2026-grounding,
  title     = {Grounding Encoders and Word Embeddings in {FST}s and Dictionaries for {U}ralic Languages},
  author    = {Alnajjar, Khalid and H{\"a}m{\"a}l{\"a}inen, Mika and Rueter, Jack and Partanen, Niko},
  booktitle = {Proceedings of the International Workshop on Computational Linguistics for Uralic Languages (IWCLUL 2026)},
  year      = {2026}
}
```

## License

The code and the models are released under the MIT License. LexiBERT is based on
[mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) (MIT).
