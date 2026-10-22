# LexiBERT demo

A small web app for trying all of the released models in the browser:

| Tab | Model | What it does |
|---|---|---|
| UD parsing | [lexibert-ud-uralic](https://huggingface.co/khalidmokha/lexibert-ud-uralic) | UPOS, features, lemmas and dependencies, drawn as a tree |
| Unknown words | [lexibert-unknown-words](https://huggingface.co/khalidmokha/lexibert-unknown-words) | lemma, part of speech, tags and inflection class of any word form |
| Masked words | [lexibert](https://huggingface.co/khalidmokha/lexibert) | the encoder itself: the most likely words for `<mask>` |
| Word embeddings | [morphemb](https://huggingface.co/khalidmokha/morphemb) | nearest words in the same or another language, similarity of two words |
| Sentiment | [lexibert-sentiment-sst2](https://huggingface.co/khalidmokha/lexibert-sentiment-sst2) | positive or negative |

Everything runs on the CPU of your own computer; no GPU is needed. Every tab has examples that can be clicked.

## What you need

- Python 3.10, 3.11, 3.12 or 3.13 and `git`
- about 8 GB of RAM and 6 GB of free disk space (the libraries and the models)
- an internet connection the first time (the models and the FSTs are downloaded once and then cached)

## 1. Get the code

```bash
git clone https://github.com/mokha/lexibert.git
cd lexibert
```

## 2. Create a Python environment

Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3. Install

On Linux, install the CPU version of PyTorch first; otherwise pip downloads the much larger GPU version (skip this
line if you have an NVIDIA GPU and want to use it, and on macOS and Windows):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the `lexibert` package and the libraries of the demo (Gradio and UralicNLP):

```bash
pip install -e .
pip install -r demo/requirements.txt
```

## 4. Run

```bash
python demo/app.py
```

When the terminal shows `Running on local URL: http://127.0.0.1:7860`, open <http://127.0.0.1:7860> in your browser.
Stop the demo with Ctrl+C.

## What to expect

- The first time a tab is used, its model is downloaded from Hugging Face (about 600 MB per model, 1.2 GB for the
  word embeddings) and loaded, which takes a minute or more. After that, the answers come in about a second.
- The FST of a language is downloaded from UralicNLP the first time the language is used.
- In the word embeddings tab, the first search in a language builds the vectors of all of its lemmas: a few seconds
  for English, Finnish and Russian, a minute or two for a Uralic language.
- The downloads are kept in the Hugging Face cache (`~/.cache/huggingface`) and in the folder of UralicNLP, so the
  next start is fast and also works offline.

## Problems

- **Port 7860 is in use:** start the demo on another port with `GRADIO_SERVER_PORT=7861 python demo/app.py`
  (PowerShell: `$env:GRADIO_SERVER_PORT=7861; python demo/app.py`).
- **Use it from another computer:** `GRADIO_SERVER_NAME=0.0.0.0 python demo/app.py`, then open
  `http://<the computer's address>:7860`.
- **`ModuleNotFoundError: No module named 'lexibert'`:** the environment is not active or step 3 was not run from the
  root of the repository; activate the environment and run `pip install -e .` again.
- **Models that are already on your computer:** set `LEXIBERT_LOCAL` to a folder that contains the model folders
  (`lexibert`, `lexibert-ud-uralic`, `lexibert-unknown-words`, `lexibert-sentiment-sst2`, `morphemb`), and nothing
  is downloaded from Hugging Face. In the Zenodo files, `lexibert-ud-uralic` and `lexibert-sentiment-sst2` are inside
  the folders `lexibert-ud` and `lexibert-sentiment-sst2`.
