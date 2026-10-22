"""LexiBERT demo: the encoder, the UD parser, the unknown word model, the sentiment classifier and MorphEmb."""
import html, os, re, threading
import gradio as gr
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

from lexibert import Parser, SentimentClassifier, UnknownWordModel
import morphemb
from morphemb import MorphEmb
from keyboard import Keyboard

torch.set_num_threads(min(4, os.cpu_count() or 2))
LOCAL = os.environ.get("LEXIBERT_LOCAL")  # a folder with the model folders, for testing without the Hub
REPOS = {"encoder": "khalidmokha/lexibert", "ud": "khalidmokha/lexibert-ud-uralic",
         "unknown": "khalidmokha/lexibert-unknown-words", "sentiment": "khalidmokha/lexibert-sentiment-sst2",
         "morphemb": "khalidmokha/morphemb", "keyboard": "khalidmokha/lexibert-keyboard"}

LANGS = {"myv": "Erzya", "mdf": "Moksha", "kpv": "Komi-Zyrian", "koi": "Komi-Permyak", "udm": "Udmurt",
         "sme": "North Sámi", "sms": "Skolt Sámi", "smn": "Inari Sámi", "olo": "Livvi", "krl": "Karelian",
         "vep": "Veps", "liv": "Livonian", "mhr": "Meadow Mari", "mrj": "Hill Mari", "yrk": "Tundra Nenets",
         "fin": "Finnish", "est": "Estonian", "eng": "English", "rus": "Russian"}
morphemb.LANG_NAME.update(LANGS)
URALIC = list(LANGS)[:15]
FST_LANGS = URALIC + ["fin", "est"]
EMB_LANGS = URALIC + ["fin", "eng", "rus"]


def choices(codes):
    return [(f"{LANGS[c]} ({c})", c) for c in codes]


# ---------------------------------------------------------------- loading (lazy, each model once)
_models, _lock = {}, threading.Lock()


def folder(key, **kw):
    if LOCAL:
        return os.path.join(LOCAL, REPOS[key].split("/")[1])
    return snapshot_download(REPOS[key], **kw)


def get(key):
    with _lock:
        if key not in _models:
            if key == "encoder":
                f = folder(key)
                _models[key] = (AutoTokenizer.from_pretrained(f), AutoModelForMaskedLM.from_pretrained(f).eval())
            elif key == "ud":
                _models[key] = Parser.from_pretrained(folder(key), device="cpu")
            elif key == "unknown":
                _models[key] = UnknownWordModel.from_pretrained(folder(key), device="cpu")
            elif key == "sentiment":
                _models[key] = SentimentClassifier.from_pretrained(folder(key), device="cpu")
            elif key == "morphemb":
                _models[key] = MorphEmb(folder(key, allow_patterns=["*.json", "*.safetensors", "lemmas/*"]), fst)
        return _models[key]


# ---------------------------------------------------------------- FSTs (only the descriptive analyser is fetched)
_fst_ready = set()


def ensure_fst(lang):
    if lang in _fst_ready:
        return
    from uralicNLP import uralicApi
    if not uralicApi.is_language_installed(lang):
        import urllib.request
        base = getattr(uralicApi, "__model_base_folders")()
        target = os.path.join(getattr(uralicApi, "__find_writable_folder")(base), lang)
        os.makedirs(target, exist_ok=True)
        urllib.request.urlretrieve(uralicApi.download_server_url + lang + "/analyser-gt-desc.hfstol",
                                   os.path.join(target, "analyser"))
    _fst_ready.add(lang)


def fst(word, lang):
    if lang not in FST_LANGS:
        return []
    ensure_fst(lang)
    from uralicNLP import uralicApi
    return [a for a, _ in uralicApi.analyze(word, lang)]


TOKEN = re.compile(r"\w+(?:[-'’ʼʹ]\w+)*|[^\w\s]", re.UNICODE)


def tokenize(text):
    return TOKEN.findall(text)


# ---------------------------------------------------------------- tabs
@torch.no_grad()
def fill_mask(text):
    tok, model = get("encoder")
    text = text.replace("[MASK]", tok.mask_token).replace("<mask>", tok.mask_token)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
    pos = (enc.input_ids[0] == tok.mask_token_id).nonzero().flatten().tolist()
    if not pos:
        raise gr.Error("Put <mask> where the missing word is.")
    probs = model(**enc).logits[0, pos[0]].softmax(-1)
    vals, ids = probs.topk(10)
    return {tok.decode([i]).strip() or repr(tok.decode([i])): float(v) for v, i in zip(vals, ids.tolist())}


def arcs_svg(sent):
    """A small dependency diagram: words on a line, arcs from the head to the dependent."""
    xs, x = [], 10
    for t in sent:
        w = max(len(t["form"]), len(t["upos"]), 4) * 8.5 + 22
        xs.append(x + w / 2); x += w
    width = x + 10
    spans = [abs(t["head"] - t["id"]) for t in sent if t["head"]]
    top = 22 + 18 * min(max(spans or [1]), 8)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{top + 52}" '
           f'style="font-family:sans-serif;font-size:13px;fill:currentColor">',
           '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
           'orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="currentColor"/></marker></defs>']
    for t, cx in zip(sent, xs):
        out.append(f'<text x="{cx:.0f}" y="{top + 22}" text-anchor="middle">{html.escape(t["form"])}</text>')
        out.append(f'<text x="{cx:.0f}" y="{top + 40}" text-anchor="middle" opacity="0.6" '
                   f'font-size="11">{html.escape(t["upos"])}</text>')
        if t["head"] == 0:
            out.append(f'<line x1="{cx:.0f}" y1="8" x2="{cx:.0f}" y2="{top + 6}" stroke="currentColor" '
                       f'marker-end="url(#a)"/><text x="{cx + 4:.0f}" y="16" font-size="11">root</text>')
            continue
        hx = xs[t["head"] - 1]
        h = 18 * min(abs(t["head"] - t["id"]), 8)
        y0 = top + 6
        out.append(f'<path d="M{hx:.0f},{y0} C{hx:.0f},{y0 - h - 10} {cx:.0f},{y0 - h - 10} {cx:.0f},{y0}" '
                   f'fill="none" stroke="currentColor" opacity="0.8" marker-end="url(#a)"/>')
        out.append(f'<text x="{(hx + cx) / 2:.0f}" y="{y0 - h - 6:.0f}" text-anchor="middle" '
                   f'font-size="11">{html.escape(t["deprel"])}</text>')
    out.append("</svg>")
    return '<div style="overflow-x:auto">' + "".join(out) + "</div>"


def parse(text, lang, use_fst):
    words = tokenize(text)
    if not words:
        raise gr.Error("Write a sentence.")
    if len(words) > 80:
        raise gr.Error("Please parse at most 80 words at a time.")
    fst_lang = None
    if use_fst and lang in FST_LANGS:
        ensure_fst(lang)
        fst_lang = lang
    sent = get("ud").parse([words], fst_lang=fst_lang)[0]
    rows = [[t["id"], t["form"], t["lemma"], t["upos"], t["feats"], t["head"], t["deprel"]] for t in sent]
    from lexibert.parser import to_conllu
    return arcs_svg(sent), rows, to_conllu([sent])


def unknown(words, lang, k):
    forms = tokenize(words)
    forms = [f for f in forms if re.search(r"\w", f)][:20]
    if not forms:
        raise gr.Error("Write one or more words.")
    res = get("unknown").analyse(lang, forms, k=int(k))
    fmt = lambda xs: ", ".join(f"{v} ({p:.2f})" for v, p in xs)
    rows = []
    for f, r in zip(forms, res):
        known = fst(f, lang)
        rows.append([f, fmt(r["lemma"]), fmt(r["pos"]), fmt(r["tags"]), fmt(r["contlex"]),
                     "; ".join(known[:4]) if known else "(not known)"])
    return rows


@torch.no_grad()
def sentiment(text):
    clf = get("sentiment")
    if not text.strip():
        raise gr.Error("Write a sentence.")
    e = clf.tok([text], truncation=True, max_length=128, return_tensors="pt")
    p = clf(e.input_ids, e.attention_mask)[0].float().softmax(-1)
    return {l: float(x) for l, x in zip(clf.labels, p)}


def neighbours(word, lang, target, n, limit, progress=gr.Progress()):
    word = word.strip()
    if not word:
        raise gr.Error("Write a word.")
    me = get("morphemb")
    found = me.analyses(word, lang)
    res = me.neighbours(word, lang, target, int(n), progress, int(limit) or None)
    note = (f"FST analyses of *{word}*: " + ", ".join(f"`{a}`" for a in found[:6])) if found else \
        (f"*{word}* is a known {LANGS[lang]} word." if lang in {"eng", "fin", "rus"} and word.lower() in me.lemma_ids[lang]
         else f"The FST does not know *{word}*, so its vector comes from its character n-grams only.")
    return [[w, s] for w, s in res], note


_keyboards = {}


def keyboard(lang):
    with _lock:
        if lang not in _keyboards:
            f = folder("keyboard", allow_patterns=["config.json", f"lexicons/{lang}.*"])
            _keyboards[lang] = Keyboard(f, lang, lambda w: fst(w, lang))
        return _keyboards[lang]


def autocorrect(text, lang):
    tokens = tokenize(text)
    if not tokens:
        raise gr.Error("Write a sentence.")
    if len(tokens) > 40:
        raise gr.Error("Please write at most 40 words at a time.")
    ensure_fst(lang)
    kb, (tok, model) = keyboard(lang), get("encoder")
    out, rows = [], []
    for t in tokens:
        if len(t) >= 3 and t.isalpha():
            best, scored = kb.correct(tok, model, " ".join(out), t)
            if best != t:
                rows.append([t, best, ", ".join(f"{c} ({s:.2f})" for c, s, *_ in scored[:4])])
            out.append(best)
        else:
            out.append(t)
    shown = " ".join(f"<b>{html.escape(o)}</b>" if o != t else html.escape(o) for o, t in zip(out, tokens))
    return f"<p style='font-size:1.1em'>{shown}</p>", rows


# the weight of spelling similarity, tuned on a sample of Ve'rdd translations (see demo/README.md)
SPELLING_WEIGHT = 0.3


def translate(word, lang, target, n, progress=gr.Progress()):
    from rapidfuzz.distance import Levenshtein
    from unidecode import unidecode
    word = word.strip()
    if not word:
        raise gr.Error("Write a word.")
    me = get("morphemb")
    m = me.matrix(target, progress)[:20000].float()
    words = me.lemma_lists[target][:20000]
    cos, idx = (m @ torch.nn.functional.normalize(me.vector(word, lang), dim=0)).topk(50)
    a = unidecode(word).lower()
    rows = []
    for c, i in zip(cos.tolist(), idx.tolist()):
        sim = Levenshtein.normalized_similarity(a, unidecode(words[i]).lower())
        rows.append([words[i], round(c, 3), round(sim, 3), round(c + SPELLING_WEIGHT * sim, 3)])
    rows.sort(key=lambda r: -r[3])
    return rows[:int(n)]


def similarity(a, la, b, lb):
    if not a.strip() or not b.strip():
        raise gr.Error("Write two words.")
    return round(get("morphemb").similarity(a.strip(), la, b.strip(), lb), 3)


# ---------------------------------------------------------------- examples (UD test sentences that are parsed fully correctly)
UD_EXAMPLES = [
    ["Весе тайгась уды кеме , плотна удомасо .", "myv"],
    ["Синь тонафнесть колмоце класса .", "mdf"],
    ["Кык чой ветлӧны детсадэ .", "kpv"],
    ["Тэ кольччы , а ме муна .", "koi"],
    ["Son ii bivdde gal akto .", "sme"],
    ["Tõt leäi nåkkam maainâs .", "sms"],
    ["Sit kolmas päivy mennäh taaste .", "olo"],
    ["Hyö tuaš vietih ne kesselin luo .", "krl"],
    ["On aina hauska saada nimille naamat .", "fin"],
    ["Muusikat võib küll proovida sõnadesse panna !", "est"],
]

ABOUT = """
# LexiBERT

Models of the paper *Grounding Encoders and Word Embeddings in FSTs and Dictionaries for Uralic Languages*
(Alnajjar, Hämäläinen, Rueter and Partanen, IWCLUL 2026). LexiBERT is mmBERT-small whose pretraining we continued on
15 Uralic languages with the help of their FSTs and the Ve'rdd dictionaries.
The first request of each tab downloads and loads its model, which takes a while.

[Encoder](https://huggingface.co/khalidmokha/lexibert) ·
[UD parser](https://huggingface.co/khalidmokha/lexibert-ud-uralic) ·
[Unknown words](https://huggingface.co/khalidmokha/lexibert-unknown-words) ·
[Sentiment](https://huggingface.co/khalidmokha/lexibert-sentiment-sst2) ·
[MorphEmb](https://huggingface.co/khalidmokha/morphemb) ·
[Code](https://github.com/mokha/lexibert)
"""

with gr.Blocks(title="LexiBERT") as demo:
    gr.Markdown(ABOUT)
    with gr.Tab("UD parsing"):
        gr.Markdown("UPOS, morphological features, lemmas and dependencies. The model was trained on the UD treebanks "
                    "of Erzya, Moksha, Komi-Zyrian, Komi-Permyak, Skolt Sámi, Livvi, Karelian, North Sámi, Veps, "
                    "Tundra Nenets, Finnish and Estonian. The text is split into words at spaces and punctuation.")
        with gr.Row():
            ud_text = gr.Textbox(label="Sentence", lines=2, scale=3)
            with gr.Column(scale=1):
                ud_lang = gr.Dropdown(choices(FST_LANGS), value="myv", label="Language")
                ud_fst = gr.Checkbox(True, label="Choose the lemma among the FST lemmas (Lem_FST)")
        ud_btn = gr.Button("Parse", variant="primary")
        ud_svg = gr.HTML()
        ud_table = gr.Dataframe(headers=["ID", "Form", "Lemma", "UPOS", "Feats", "Head", "Deprel"], wrap=True)
        ud_conllu = gr.Code(label="CoNLL-U")
        gr.Examples(UD_EXAMPLES, [ud_text, ud_lang], label="Examples (UD test sentences)")
        ud_btn.click(parse, [ud_text, ud_lang, ud_fst], [ud_svg, ud_table, ud_conllu])
        ud_text.submit(parse, [ud_text, ud_lang, ud_fst], [ud_svg, ud_table, ud_conllu])

    with gr.Tab("Unknown words"):
        gr.Markdown("The lemma, part of speech, tags and inflection class (contlex) of word forms, also of words that "
                    "the FST does not know. The last column shows what the FST says, for comparison.")
        with gr.Row():
            uw_text = gr.Textbox(label="Word forms (separated by spaces)", scale=3)
            uw_lang = gr.Dropdown(choices(URALIC), value="myv", label="Language", scale=1)
            uw_k = gr.Slider(1, 10, 3, step=1, label="Candidates", scale=1)
        uw_btn = gr.Button("Analyse", variant="primary")
        uw_table = gr.Dataframe(headers=["Form", "Lemma", "POS", "Tags", "Contlex", "FST"], wrap=True)
        gr.Examples([["кудосо", "myv"], ["токсесазь кядьса", "mdf"], ["семяын", "kpv"], ["vástidii", "sme"],
                     ["pidäy", "olo"]], [uw_text, uw_lang])
        uw_btn.click(unknown, [uw_text, uw_lang, uw_k], uw_table)
        uw_text.submit(unknown, [uw_text, uw_lang, uw_k], uw_table)

    with gr.Tab("Masked words"):
        gr.Markdown("The encoder itself: write `<mask>` in place of a word.")
        fm_text = gr.Textbox(label="Text", lines=2)
        fm_btn = gr.Button("Predict", variant="primary")
        fm_out = gr.Label(num_top_classes=10, label="Most likely tokens")
        gr.Examples([["Kirilällä oli šiih aikah nellä <mask> ."], ["– Iče <mask> karjalaine Kiihtelysvuaraspäi ."],
                     ["Mun <mask> láibumin gáhkuid ."], ["Медводзсэ ме помалі Красноборса шӧр <mask> ."],
                     ["Vaiše lebupäivile <mask> voiba ajada kodihe , vastatas vanhembidenke ."],
                     ["Toinen tärkeä <mask> on keskusteleminen ."]], [fm_text],
                    label="Examples (UD sentences; the masked words are vuotta, olen, lean, школа, lapsed and asia)")
        fm_btn.click(fill_mask, fm_text, fm_out)
        fm_text.submit(fill_mask, fm_text, fm_out)

    with gr.Tab("Word embeddings"):
        gr.Markdown("MorphEmb puts 15 Uralic languages, Finnish, English and Russian in one vector space. The vector of "
                    "a word is built from its character n-grams and its FST analyses, so any word form gets one. "
                    "The neighbours are searched among the lemmas of the chosen language and are mostly other forms "
                    "and compounds of the same word. Searching in another language sometimes finds the translation, as in "
                    "the examples below, where only the most frequent words are searched. For a "
                    "Uralic language, the lemma vectors are built the first time, which takes a minute or two.")
        with gr.Row():
            me_word = gr.Textbox(label="Word", scale=2)
            me_lang = gr.Dropdown(choices(EMB_LANGS), value="myv", label="Language of the word", scale=1)
            me_target = gr.Dropdown(choices(EMB_LANGS), value="myv", label="Search in", scale=1)
            me_n = gr.Slider(1, 30, 10, step=1, label="Neighbours", scale=1)
            me_limit = gr.Slider(0, 100000, 20000, step=5000, label="Most frequent words only (0 = all)", scale=1)
        me_btn = gr.Button("Find neighbours", variant="primary")
        me_note = gr.Markdown()
        me_table = gr.Dataframe(headers=["Word", "Cosine"])
        gr.Examples([["кудо", "myv", "myv"], ["кудосо", "myv", "myv"], ["йылме", "mhr", "fin"], ["самолёт", "kpv", "fin"],
                     ["ođastus", "sme", "fin"], ["мунны", "koi", "fin"], ["koir", "vep", "fin"], ["тӧдны", "kpv", "eng"],
                     ["квартира", "kpv", "eng"], ["школа", "yrk", "eng"], ["vähitellen", "olo", "rus"],
                     ["radnik", "vep", "rus"], ["mies", "olo", "rus"]],
                    [me_word, me_lang, me_target], label="Examples (the first result is a dictionary translation)")
        me_btn.click(neighbours, [me_word, me_lang, me_target, me_n, me_limit], [me_table, me_note])
        gr.Markdown("### Similarity of two words")
        with gr.Row():
            s_a = gr.Textbox(label="First word"); s_la = gr.Dropdown(choices(EMB_LANGS), value="myv", label="Language")
            s_b = gr.Textbox(label="Second word"); s_lb = gr.Dropdown(choices(EMB_LANGS), value="fin", label="Language")
        s_btn = gr.Button("Compare")
        s_out = gr.Number(label="Cosine similarity")
        gr.Examples([["кудо", "myv", "talo", "fin"], ["тӧдны", "kpv", "know", "eng"]], [s_a, s_la, s_b, s_lb])
        s_btn.click(similarity, [s_a, s_la, s_b, s_lb], s_out)

    with gr.Tab("Keyboard"):
        gr.Markdown("Autocorrection for text typed on a Finnish or Russian keyboard, where the special letters of the "
                    "language are missing or a key next to the right one was pressed. The candidates of each word are "
                    "its versions with the missing letters restored and the words within one edit that occur in our "
                    "corpora or that the FST accepts, and they are ranked by \\enc, the corpus frequency and the "
                    "number of edits. Changed words are shown in bold.".replace("\\enc", "LexiBERT"))
        with gr.Row():
            kb_text = gr.Textbox(label="Typed sentence", lines=2, scale=3)
            kb_lang = gr.Dropdown(choices(URALIC), value="kpv", label="Language", scale=1)
        kb_btn = gr.Button("Correct", variant="primary")
        kb_out = gr.HTML()
        kb_table = gr.Dataframe(headers=["Typed", "Corrected", "Best candidates (score)"], wrap=True)
        KB_EXAMPLES = [["Медборын лэччылі Нярьян-Маре тоже оленеводствоо , Нарьян-Марскей ОПХ-а .", "kpv"],
                       ["Ютасть мазый шканзо лемзер чувтыненть .", "myv"],
                       ["Козя ся колхозсь и сероса , и ярмакса .", "mdf"],
                       ["Ferte mearridit makkar deattu dáidda galgá bidjat .", "sme"],
                       ["– Ice olen karjalaine Kiihtelysvuaraspäi .", "olo"],
                       ["Hyö tultih suon laitah , heitettih kesseli tiellä .", "krl"],
                       ["De suu vaaldi kaavân tõt ooumaž , son vuõʹlji .", "sms"],
                       ["Nece rad pakiceb vastusenpidändad da väged .", "vep"]]
        if KB_EXAMPLES:
            gr.Examples(KB_EXAMPLES, [kb_text, kb_lang], label="Examples (UD test sentences with the special letters of one word missing)")
        kb_btn.click(autocorrect, [kb_text, kb_lang], [kb_out, kb_table])
        kb_text.submit(autocorrect, [kb_text, kb_lang], [kb_out, kb_table])

    with gr.Tab("Translations"):
        gr.Markdown("Translation suggestions for adding to a dictionary: the words of the target language that are "
                    "closest to the word in MorphEmb, reranked with their spelling similarity, which helps with "
                    "loanwords and related languages. **Limitation:** in the paper, the suggestions come from the "
                    "translation graph of the Ve'rdd dictionary, a ranker and an LLM. The dictionary data is not "
                    "ours to distribute and the LLM is too large for this demo, so the demo uses only the word "
                    "embeddings and gives weaker suggestions than the method of the paper.")
        with gr.Row():
            tr_word = gr.Textbox(label="Word", scale=2)
            tr_lang = gr.Dropdown(choices(URALIC), value="myv", label="Language of the word", scale=1)
            tr_target = gr.Dropdown(choices(["fin", "eng"]), value="fin", label="Translate into", scale=1)
            tr_n = gr.Slider(1, 20, 10, step=1, label="Suggestions", scale=1)
        tr_btn = gr.Button("Suggest translations", variant="primary")
        tr_table = gr.Dataframe(headers=["Suggestion", "Cosine", "Spelling", "Score"])
        TR_EXAMPLES = [["naini", "krl", "fin"], ["vägivaldu", "olo", "fin"], ["kučču", "krl", "fin"],
                       ["kontsert", "liv", "fin"], ["vaattâđ", "smn", "fin"], ["energiija", "sme", "fin"],
                       ["заводитны", "kpv", "eng"], ["дружба", "kpv", "eng"], ["договор", "myv", "fin"],
                       ["преподаватель", "mrj", "fin"], ["ритм", "mhr", "fin"]]
        if TR_EXAMPLES:
            gr.Examples(TR_EXAMPLES, [tr_word, tr_lang, tr_target],
                        label="Examples from Ve'rdd (the first suggestion is the dictionary translation)")
        tr_btn.click(translate, [tr_word, tr_lang, tr_target, tr_n], tr_table)

    with gr.Tab("Sentiment"):
        gr.Markdown("A classifier trained on English SST-2 only. In the paper it was tested on Erzya, Moksha, "
                    "Komi-Zyrian, Udmurt and Finnish without any training data in those languages.")
        se_text = gr.Textbox(label="Sentence", lines=2)
        se_btn = gr.Button("Classify", variant="primary")
        se_out = gr.Label(label="Sentiment")
        gr.Examples([["This movie was wonderful."], ["I hated every minute of it."]], [se_text])
        se_btn.click(sentiment, se_text, se_out)
        se_text.submit(sentiment, se_text, se_out)

if __name__ == "__main__":
    import pyhfst
    print("FST lookups: pyhfst,", "compiled (fast)" if getattr(pyhfst, "CYTHON", False) else
          "pure Python (slower; see demo/README.md for compiling it)", flush=True)
    demo.queue(default_concurrency_limit=2).launch()
