#!/usr/bin/env bash
. ../exp/data_path.sh
FILE=${1:-"test_results/s2t_ctc_asr_1/char_score.en"}
NUM=${2:-500}
SET=${3:-"test"}
OUT=${4:-"test2"}
DIR=./data
BEST=$DIR/$SET.best_id

# get list of best wer line numbers
grep -E "SENTENCE|Errors" $FILE | \
    awk '/[0-9]$/ { printf("%s\t", $0); next } 1' | \
    sed 's/\%//g' | \
    sort -k5 -n | \
    head -$NUM | \
    cut -f1 | \
    cut -d' ' -f2 > $BEST

# get these examples as test2 set
for suf in wav_list $SRC $TGT; do
    awk 'NR==FNR{ a[$1]; next }FNR in a' $BEST $DIR/$SET.$suf > $DIR/$OUT.$suf
done