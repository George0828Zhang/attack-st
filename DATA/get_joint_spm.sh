SRC=en
TGT=${1:-zh-CN}
DATA_ROOT=/livingrooms/george/covost2-atk

FAIRSEQ=~/utility/fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
source ~/envs/spml/bin/activate
SPM_TRAIN=${FAIRSEQ}/scripts/spm_train.py
OUTDIR=${DATA_ROOT}/${SRC}

tmpfile=$(mktemp /tmp/abc-script.XXXXXX)

cd ${OUTDIR}
cut -f4 train_st_${SRC}_${TGT}.tsv > ${tmpfile}
SPM_PREFIX=spm_char_st_${SRC}
SPM_MODEL=${SPM_PREFIX}.model
DICT=${SPM_PREFIX}.txt

if [ -f $SPM_MODEL ]; then
    echo "SPM model: $SPM_MODEL exists, skip learning"
else
    ccvg=1.0
    # if [ ${lang} = "zh-CN" ]; then
    #     ccvg=0.9995
    # fi
    python ${SPM_TRAIN} --input=${tmpfile} \
        --model_prefix=$SPM_PREFIX \
        --character_coverage=${ccvg} \
        --model_type=char
    cut -f1 ${SPM_PREFIX}.vocab | tail -n +4 | sed "s/$/ 100/g" > ${DICT}
fi

rm "$tmpfile"