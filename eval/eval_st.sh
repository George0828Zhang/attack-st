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
    -b|--beam)
      BEAM="$2"
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
MODEL=${MODEL:-s2t_st_load5_sd1}
BEAM=${BEAM:-5}
SRC_FILE=${SRC_FILE:-data/test.wav_list}
TGT_FILE=${TGT_FILE:-data/test.zh-CN}
EXP=${EXP:-../exp}
. ${EXP}/data_path.sh
CONF=$DATA/config_st_${SRC}_${TGT}.yaml
CHECKDIR=${EXP}/checkpoints/${MODEL}
RESULTS=$(dirname ${SRC_FILE})_results/${MODEL}
AVG=true

EXTRAARGS="--beam ${BEAM}
--max-len-a 0.1
--max-len-b 10
--post-process sentencepiece
"

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

mkdir -p ${RESULTS}

lang=${TGT}

cp ${TGT_FILE} ${RESULTS}/refs.${lang}

cat ${SRC_FILE} | \
python -m fairseq_cli.interactive ${DATA} --user-dir ${USERDIR} \
    --config-yaml ${CONF} \
    --task speech_to_text_infer \
    --buffer-size 4096 --batch-size 64 \
    --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
    --model-overrides '{"load_pretrained_encoder_from": None}' \
    ${EXTRAARGS} | \
grep -E "D-[0-9]+" | \
cut -f3 > ${RESULTS}/hyps.${lang}

echo "# evaluating BLEU"
python -m sacrebleu ${RESULTS}/refs.${lang} \
    -i ${RESULTS}/hyps.${lang} \
    -m bleu \
    --width 2 \
    -tok zh | tee ${RESULTS}/score.${lang}
