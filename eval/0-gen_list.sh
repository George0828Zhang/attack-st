#!/usr/bin/env bash
SPLIT=dev
EVAL_DATA=./data
. ../exp/data_path.sh

echo "segmenting ${SPLIT} data"
python ../DATA/seg_covost_data.py \
  --data-root ${DATA_ROOT} -s ${SRC} -t ${TGT} \
  --split ${SPLIT} \
  --max-instance 500 \
  --max-frames 6000 \
  --output ${EVAL_DATA}
