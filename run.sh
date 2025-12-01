#!/bin/bash
#SBATCH --time=01:00
#SBATCH --account=deep_learning
. /etc/profile.d/modules.sh
module add cuda/12.9
source .venv/bin/activate
python3 src/main.py +experiment=miditok