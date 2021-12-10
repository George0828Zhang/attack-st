export SRC=en
export TGT=zh-CN
export DATA_ROOT=/livingrooms/george/covost2-atk
export DATA=${DATA_ROOT}/${SRC}

FAIRSEQ=`realpath ~/utility/fairseq`
USERDIR=`realpath ../code_base`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
. ~/envs/spml/bin/activate

export NUMEXPR_MAX_THREADS=4