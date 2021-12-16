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
    -s|--split)
      SPLIT="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--out)
      OUT="$2"
      shift # past argument
      shift # past value
      ;;
    -n|--num)
      NUM="$2"
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
EXP=${EXP:-../exp}
. ${EXP}/data_path.sh
FILE=${FILE:-"test.wav_list_results/s2t_ctc_asr_1/char_score.en"}
NUM=${NUM:-500}
SPLIT=${SPLIT:-"test"}
OUT=${OUT:-"benign"}
BEST=${SPLIT}.best_id

# get list of best wer line numbers
grep -E "SENTENCE|Errors" $FILE | \
    awk '/[0-9]$/ { printf("%s\t", $0); next } 1' | \
    sed 's/\%//g' | \
    sort -k5 -n | \
    head -$NUM | \
    cut -f1 | \
    cut -d' ' -f2 > $BEST

# get these examples as test2 set
mkdir -p $OUT
for suf in wav_list $SRC $TGT; do
    awk 'NR==FNR{ a[$1]; next }FNR in a' $BEST data/$SPLIT.$suf > $OUT/test.$suf
done