#!/usr/bin/env bash
SEED=${1:-1}
TASK=s2t_ctc_asr_${SEED}
. ./data_path.sh
CHECKASR=/home/george/simulst/exp/checkpoints/s2t_ctc_asr/checkpoint_last.pt

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-encoder-from ${CHECKASR} \
    --config-yaml config_st_${SRC}_${TGT}.yaml \
    --train-subset train_st_${SRC}_${TGT} \
    --valid-subset dev_st_${SRC}_${TGT} \
    --max-tokens 160000 \
    --update-freq 2 \
    --task speech_to_text_infer --do-asr \
    --inference-config-yaml infer_asr.yaml \
    --arch s2t_speech_encoder_s \
    --criterion label_smoothed_ctc --label-smoothing 0.1 --report-accuracy \
    --clip-norm 10.0 \
    --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --max-update 300000 \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --wandb-project attack-st \
    --best-checkpoint-metric wer \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 5 \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 4 \
    --fp16 \
    --seed ${SEED}
