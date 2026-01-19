# DL 2025 Project

> by Sebastian Brunner, Felix Schatzl, Harald Semmelrock


# Get Started

- Login on student cluster via `<user>@student-cluster.inf.ethz.ch`
- Clone the project repo into your home directory
- Setup `venv`
    - `python3 -m venv .venv` (*make sure it's called* `.venv` cause the run script will use this)
    - Enable the `.venv`: `source .venv/bin/activate`
    - Your prompt should now show `(.venv)` at the start of the line
    - [PyTorch & Cuda Guide](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterCuda)
- Install dependencies
  - `pip install -r requirements.txt`
  - Or manually (since torch/pip is sometimes tricky):
      - `pip install torch torchvision`
      - `pip install miditok hydra-core lightning wandb note-seq frechet-music-distance muspy`
- Login to WandB
    - `wandb login <key>`
- Run jobs  (on cluster)
  - Training: `sbatch run.sh <experiment name here>`
  - Evaluation: `sbatch eval.sh <experiment name here> <path/to/some.ckpt>`
  - [Running jobs](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobs)
- Local/dev runs (quick runs for development)
  - Training: `python src/main.py +experiment=<experiment name here> +run=dev`
  - Evaluation: `python src/eval.py +experiment=<experiment name here> +run=dev +checkpoint=\"<path/to/some.ckpt>\"`

**Note:** The best checkpoint of each experiment will be saved in `./outputs/checkpoints/<experiment_name_here>/best.ckpt`, if you already want to schedule an evaluation before training completes.
This file will be overridden by experiments with the same name tho.

# Experiments

# Repository Structure

Important files, directories:

- `cfg/`: config files for [hydra](https://hydra.cc/docs/intro/), we configure and instantiate models, classes, etc.
  - `dataset/*`: `lakh-MIDI-1k` is a smaller subset of `lakh-MIDI-10k` which we use for training/evaluation
  - `experiment/*`: see [Experiments](#experiments)
  - `run/*`: run configurations for different environments, `dev` = quick runs for development, `cluster` = runs on cluster
  - `config.yaml`: basic configuration parameters for all runs
- `src/`: Python files/code
  - `dataset`