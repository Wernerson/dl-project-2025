# DL 2025

> by Harald Semmelrock, Felix Schatzl, Sebastian Brunner


## How to...

### Setup the cluster

- [PyTorch & Cuda Guide](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterCuda)
    - Furthermore, after `venv` creation and activation, run:
    - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --upgrade`
    - `pip install miditok hydra-core lightning wandb`
- [Running jobs](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobs)

### Add an external library / repo

1. `cd src/libs`
2. `git submodule add --name <name> -f <repo_url> ./<name>`
3. add `<name>` to `libs` in `cfg/config.yaml`
4. import library in code with `libs.<name>`

### Run the training

```bash
python src/main.py +experiment=<my_experiment>
```