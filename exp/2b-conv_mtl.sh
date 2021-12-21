#!/usr/bin/env bash
# credits: https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -a|--asr)
      CHECKASR="--load-pretrained-encoder-from $2"
      F3="load$(echo $2 | sed 's/.*s2t_ctc_asr_\([0-9]\).*$/\1/')"
      shift # past argument
      shift # past value
      ;;
    -m|--mtl)
      CRITERION="--criterion label_smoothed_mtl --asr-factor 0.3 --report-accuracy"
      F2="mtl"
      shift # past argument
      ;;
    -s|--seed)
      SEED="--seed $2"
      F4="sd$2"
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
CRITERION=${CRITERION:-"--criterion label_smoothed_cross_entropy"}
SEED=${SEED:-"--seed 1"}
F2=${F2:-"st"}
F3=${F3:-"scratch"}
F4=${F4:-"sd1"}

TASK=conv_${F2}_${F3}_${F4}
. ./data_path.sh

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    ${CHECKASR} ${SEED} \
    --config-yaml config_st_${SRC}_${TGT}.yaml \
    --train-subset train_st_${SRC}_${TGT} \
    --valid-subset dev_st_${SRC}_${TGT} \
    --max-tokens 80000 \
    --update-freq 4 \
    --task speech_to_text_infer \
    --inference-config-yaml infer_st.yaml \
    --arch conv_seq2seq_s \
    ${CRITERION} --label-smoothing 0.1 \
    --clip-norm 1.0 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr 2e-3 --lr-scheduler inverse_sqrt \
    --dropout 0.15 --warmup-init-lr 1e-7 --weight-decay 0.01 \
    --warmup-updates 10000 \
    --max-update 110000 \
    --save-dir checkpoints/${TASK} \
    --wandb-project attack-st \
    --no-epoch-checkpoints \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 5 \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 4 \
    --fp16
    
