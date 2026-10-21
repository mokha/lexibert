"""Download only the FST / CG models from the UralicNLP server, in parallel."""
import os, urllib.request, ssl
from concurrent.futures import ThreadPoolExecutor
from uralicNLP import uralicApi

ssl._create_default_https_context = ssl._create_unverified_context
LANGS = ["myv", "mdf", "kpv", "koi", "udm", "sme", "sms", "smn", "olo", "krl", "vep",
         "liv", "mhr", "mrj", "yrk", "mns", "fin", "est"]
FILES = {"analyser": "analyser-gt-desc.hfstol", "analyser-norm": "analyser-gt-norm.hfstol",
         "generator-desc": "generator-gt-desc.hfstol", "generator-norm": "generator-gt-norm.hfstol",
         "cg": "disambiguator.bin", "metadata.json": "metadata.json"}
base = getattr(uralicApi, "__find_writable_folder")(getattr(uralicApi, "__model_base_folders")())
server = uralicApi.download_server_url


def get(job):
    lang, name, remote = job
    os.makedirs(os.path.join(base, lang), exist_ok=True)
    dst = os.path.join(base, lang, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return f"have {lang}/{name}"
    try:
        urllib.request.urlretrieve(server + lang + "/" + remote, dst)
        return f"ok {lang}/{name} {os.path.getsize(dst)}"
    except Exception as e:
        if os.path.exists(dst):
            os.remove(dst)
        return f"MISSING {lang}/{name} {e}"


with ThreadPoolExecutor(16) as ex:
    for r in ex.map(get, [(l, n, r) for l in LANGS for n, r in FILES.items()]):
        print(r, flush=True)
print("base", base)
