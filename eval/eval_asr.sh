#!/usr/bin/env bash
# credits: https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -m|--model)
      MODEL="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--source)
      SRC_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--target)
      TGT_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -e|--exp)
      EXP="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

# defaults
MODEL=${MODEL:-s2t_ctc_asr_1}
SRC_FILE=${SRC_FILE:-data/test.wav_list}
TGT_FILE=${TGT_FILE:-data/test.en}
EXP=${EXP:-../exp}
. ${EXP}/data_path.sh
CONF=$DATA/config_asr.yaml
CHECKDIR=${EXP}/checkpoints/${MODEL}
RESULTS=$(dirname ${SRC_FILE})_results/${MODEL}
AVG=true

EXTRAARGS=""

if [[ $AVG == "true" ]]; then
    CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
    if [ ! -f ${CHECKDIR}/${CHECKPOINT_FILENAME} ]; then
      python ../scripts/average_checkpoints.py \
        --inputs ${CHECKDIR} --num-best-checkpoints 5 \
        --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
    fi
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

cp ${TGT_FILE} ${RESULTS}/refs.${lang}

cat ${SRC_FILE} | \
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
# echo "# evaluating WER"
# wer -a ${RESULTS}/refs.${lang} ${RESULTS}/hyps.${lang} | tee ${RESULTS}/score.${lang}
echo "# evaluating CER"
wer -i -a <(char ${RESULTS}/refs.${lang}) <(char ${RESULTS}/hyps.${lang}) | tee ${RESULTS}/char_score.${lang}
