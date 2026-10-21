#!/bin/bash
# The experiments of the paper in order. Every step reads and writes under $LEXIBERT_WORK (default: ./work).
# The data are not distributed: steps 00-02 need your own copy of the Ve'rdd database and the corpora (see README).
set -e
export LEXIBERT_WORK=${LEXIBERT_WORK:-work} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="python -W ignore"
L="myv mdf kpv koi udm sme sms smn olo krl vep liv mhr mrj yrk"

# 1. data: UD treebanks, FSTs, FineWeb2, pivot-language text, Ve'rdd, corpora, FST analyses, pretraining lookups
$PY scripts/00_fetch.py
$PY scripts/00b_fetch_fsts.py
$PY scripts/00c_fetch_pivot.py
$PY scripts/01_export_verdd.py          # needs VERDD_USER and VERDD_PASSWORD
$PY scripts/02_build_corpus.py
$PY scripts/03_analyze_types.py $L
$PY scripts/04_build_oov_data.py $L
$PY scripts/05_prep_pretrain.py
$PY scripts/27_prep_gloss.py

# 2. continued pretraining (ablation table): MLM, +FST, +FST+CS+pairs, gloss alignment, gloss pairs (LexiBERT)
$PY scripts/06_pretrain.py --name pt_mlm --steps 6000
$PY scripts/06_pretrain.py --name pt_fst --fst --steps 6000
$PY scripts/06_pretrain.py --name pt_full --fst --cs --par --steps 6000
$PY scripts/06_pretrain.py --name pt_align --align --steps 6000 --keep_last
$PY scripts/06_pretrain.py --name pt_gloss --gloss --steps 6000 --keep_last

# 3. evaluation of each encoder: UD parsing, unknown words, sentiment and retrieval,
#    keyboard autocorrection
$PY scripts/16_autocorrect.py candidates
evaluate() {  # name encoder
  for s in 1 2 3; do for tb in myv sme; do
    $PY scripts/07_ud.py --encoder $2 --name ${1}_$tb --setting mono --train $tb --steps 2000 --seed $s
  done; done
  $PY scripts/07_ud.py --encoder $2 --name $1 --setting multi --steps 6000 --seed 1
  $PY scripts/10_oov_model.py --encoder $2 --name $1 --steps 4000
  $PY scripts/14_downstream.py --encoder $2 --name $1 --seeds 1 2 3
  $PY scripts/16_autocorrect.py score --encoder $2 --name $1
}
evaluate mmbert_base jhu-clsp/mmBERT-small
for N in pt_mlm pt_fst pt_full pt_align pt_gloss; do evaluate $N $LEXIBERT_WORK/models/$N/hf; done
$PY scripts/23_reeval.py ud
$PY scripts/24_fst_joint.py
$PY scripts/26_sentiment_gloss.py mmbert_base pt_mlm pt_fst pt_full pt_align pt_gloss
$PY scripts/08_stanza_eval.py

# 4. MorphEmb from LexiBERT: without and with the unknown word hypotheses
E=$LEXIBERT_WORK/models/pt_gloss/hf
$PY scripts/11_morph_embeddings.py --encoder $E --name me_pt_gloss_nohyp --stages teacher train
$PY scripts/15_oov_hyp.py pt_gloss
mkdir -p $LEXIBERT_WORK/emb/me_pt_gloss && cp $LEXIBERT_WORK/emb/me_pt_gloss_nohyp/teacher_*.pt $LEXIBERT_WORK/emb/me_pt_gloss/
ME_HYP=$LEXIBERT_WORK/data/oov_hyp/pt_gloss $PY scripts/11_morph_embeddings.py --encoder $E --name me_pt_gloss --stages train
for M in me_pt_gloss_nohyp me_pt_gloss; do $PY scripts/14_downstream.py --emb $M --name $M; done

# 5. static word embeddings in the BiLSTM parser
$PY scripts/12_fasttext_baseline.py $L
static() {  # vectors name
  for s in 1 2 3; do for tb in myv sme; do
    $PY scripts/13_ud_static.py --vectors $1 --name ${2}_$tb --setting mono --train $tb --seed $s
  done; done
  $PY scripts/13_ud_static.py --vectors $1 --name $2 --setting multi --steps 8000 --seed 1
}
for V in none stanza a2023 fasttext; do static $V $V; done
for M in me_pt_gloss_nohyp me_pt_gloss; do static morph:$M $M; done
$PY scripts/23_reeval.py ud_static

# 6. translation prediction: graph ranker, + MorphEmb candidates, + LLM reranking
$PY scripts/09_linkpred.py
$PY scripts/09_linkpred.py --emb me_pt_gloss --name me_pt_gloss
LP_DUMP=$LEXIBERT_WORK/runs_llm/lp_dump_gloss.json $PY scripts/09_linkpred.py --emb me_pt_gloss --name llm_dump_gloss
# start an OpenAI-compatible server with Qwen3.8-27B (e.g. llama.cpp llama-server) first
$PY scripts/25_llm_rerank.py --url http://localhost:8091/v1 --split dev --dump lp_dump_gloss.json --tag _gloss
$PY scripts/25_llm_rerank.py --url http://localhost:8091/v1 --split test --dump lp_dump_gloss.json --tag _gloss

# 7. tables and numbers of the paper
$PY scripts/20_tables_resources.py
$PY scripts/21_tables_results.py
$PY scripts/22_numbers.py
