#!/usr/bin/env bash
SRC=en
TGT=${1:-zh-CN}
DATA_ROOT=/media/george/Data/covost2
vocab=5000  # actually unused for char
vtype=char
WORKERS=1
EXTRA=""

FAIRSEQ=~/utility/fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
# source ~/envs/apex/bin/activate

OUTDIR=${DATA_ROOT}/${SRC}

# ST
feats=${OUTDIR}/fbank80.zip
if [ -f ${feats} ]; then
  echo "${feats} already exists. It is likely that you set the wrong language which is already processed."
  echo "Please change data root or clear ${feats} before continuing."
  echo "Alternatively uncomment the command below to re-process manifest only."
  python prep_covost_data.py \
    --data-root ${DATA_ROOT} --vocab-type $vtype --vocab-size $vocab --joint-dict \
    --src-lang $SRC --tgt-lang $TGT --manifest-only ${EXTRA}
else
  echo "processing ${OUTDIR}"
  python prep_covost_data.py \
    --data-root ${DATA_ROOT} --vocab-type $vtype --vocab-size $vocab --joint-dict \
    --src-lang $SRC --tgt-lang $TGT ${EXTRA} \
    --cmvn-type utterance
fi
