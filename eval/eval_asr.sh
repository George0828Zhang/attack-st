#!/usr/bin/env bash
MODEL=${1:-s2t_ctc_asr_1}
SPLIT=${2:-dev}
EXP=../exp
. ${EXP}/data_path.sh
# DATA=${DATA_ROOT}/joint
CONF=$DATA/config_asr.yaml
CHECKDIR=${EXP}/checkpoints/${MODEL}
RESULTS=${SPLIT}_results/${MODEL}
AVG=true

EXTRAARGS=""

if [[ $AVG == "true" ]]; then
    CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
    # python ../scripts/average_checkpoints.py \
    #   --inputs ${CHECKDIR} --num-best-checkpoints 5 \
    #   --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
else
    CHECKPOINT_FILENAME=checkpoint_best.pt
fi

function char (){
    sed -e 's/./& /g' -e "s/[[:punct:]]\+//g" -e 's/ \{2,\}/ /g' $1
}

mkdir -p ${RESULTS}

lang=${SRC}
# tsv=${DATA}/${SPLIT}_st_${SRC}_${TGT}.tsv
# tail +2 ${tsv} | cut -f2 > ${RESULTS}/feats.${lang}
# tail +2 ${tsv} | cut -f4 > ${RESULTS}/refs.${lang}
# cat ${RESULTS}/feats.${lang} | \

cp ./data/${SPLIT}.en ${RESULTS}/refs.${lang}

cat ./data/${SPLIT}.wav_list | \
python -m fairseq_cli.interactive ${DATA} --user-dir ${USERDIR} \
    --config-yaml ${CONF} \
    --gen-subset ${SPLIT}_st_${SRC}_${TGT} \
    --task speech_to_text_infer --do-asr \
    --buffer-size 4096 --batch-size 128 \
    --inference-config-yaml infer_asr.yaml \
    --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
    --model-overrides '{"load_pretrained_encoder_from": None}' \
    ${EXTRAARGS} | \
grep -E "D-[0-9]+" | \
cut -f3 > ${RESULTS}/hyps.${lang}
echo "# evaluating WER"
wer -a ${RESULTS}/refs.${lang} ${RESULTS}/hyps.${lang} | tee ${RESULTS}/score.${lang}
echo "# evaluating CER"
wer -i -a <(char ${RESULTS}/refs.${lang}) <(char ${RESULTS}/hyps.${lang}) | tee ${RESULTS}/char_score.${lang}
