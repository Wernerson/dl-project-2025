#!/bin/bash
#SBATCH --time=180
#SBATCH --account=deep_learning
#SBATCH --gpus 5060ti:1
#SBATCH --mem=24G

if [ -z "$1" ]; then
    echo "You must pass an experiment, e.g. 'miditok'"
    exit 1
fi

if [ -z "$2" ]; then
    echo "You must pass a checkpiont, e.g. 'outputs/.../last.ckpt'"
    exit 1
fi

. /etc/profile.d/modules.sh
module add cuda/12.9
source .venv/bin/activate
export HF_HOME="/work/scratch/$USER/.cache/huggingface/datasets"
export HYDRA_FULL_ERROR=1
python3 src/main.py +experiment=$1 +run=cluster
python3 src/eval.py +experiment=$1 +run=cluster "+checkpoint=\"$2\""