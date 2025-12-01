#!/bin/bash
#SBATCH --time=00:59
#SBATCH --account=deep_learning
if [ -z "$1" ]; then
    echo "You must pass an experiment, e.g. 'miditok'"
    exit 1
fi
. /etc/profile.d/modules.sh
module add cuda/12.9
source .venv/bin/activate
python3 src/main.py +experiment=$1 +run=cluster