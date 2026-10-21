# LexiBERT demo

A small web app for trying the released models: UD parsing, unknown words, masked words, MorphEmb word embeddings and
sentiment. The models are downloaded from Hugging Face on first use and run on the CPU.

```bash
pip install -e .                   # the lexibert package, from the root of this repository
pip install -r demo/requirements.txt
python demo/app.py                 # then open http://127.0.0.1:7860
```

The FSTs of UralicNLP are downloaded the first time a language is used. For a Uralic language, the first search of the
word embeddings in that language builds the vectors of all of its lemmas, which takes a minute or two.
