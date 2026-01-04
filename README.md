# DL 2025

> by Harald Semmelrock, Felix Schatzl, Sebastian Brunner


## How to...

### Setup & use the cluster

- Login on student cluster via `<user>@student-cluster.inf.ethz.ch`
- Clone the project repo into your home directory
- Setup `venv`
    - `python -m venv .venv` (make sure it's called `.venv` cause the run script will use this)
    - [PyTorch & Cuda Guide](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterCuda)
- Install `torch` and other project dependencies
    - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --upgrade`
    - `pip install miditok hydra-core lightning wandb note-seq frechet-music-distance`
- Login to WandB
    - `wandb login <key>`
- Run jobs with: `sbatch run.sh <experiment name here>`
    - [Running jobs](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobs)

### Add an external library / repo

1. `cd src/libs`
2. `git submodule add --name <name> -f <repo_url> ./<name>`
3. add `<name>` to `libs` in `cfg/config.yaml`
4. import library in code with `libs.<name>`

### Run the training

```bash
python src/main.py +experiment=<my_experiment> +run=dev
```

Note that `+run=dev` uses the `run/dev.yaml` config, which does not log to WandB and only trains 1 epoch (for quick
runs).