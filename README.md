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
    - Note that this might take hours
    - Maybe adjust the parameters in `cfg/config.yaml`
    - e.g., `max_epochs: 1` or `dataset: lakh-midi-1k`
  - Evaluation: `sbatch eval.sh <experiment name here> <path/to/some.ckpt>`
    - Note that this might take minutes
    - Maybe adjust the number of samples in `cfg/config.yaml`
    - `num_samples: 10000` under `evaluator:`
  - [Running jobs](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobs)
- Local/dev runs (quick runs for development)
  - Training: `python src/main.py +experiment=<experiment name here> +run=dev`
  - Evaluation: `python src/eval.py +experiment=<experiment name here> +run=dev +checkpoint=\"<path/to/some.ckpt>\"`

**Note:** The best checkpoint of each experiment will be saved in `./outputs/checkpoints/<experiment_name_here>/best.ckpt`, if you already want to schedule an evaluation before training completes.
This file will be overridden by experiments with the same name tho.

# Samples

Here are two samples generated from one of our models:

<audio controls>
  <source src="samples/sample_1.wav" type="audio/mpeg">
</audio>

<audio controls>
  <source src="samples/sample_1.mp4" type="audio/mpeg">
</audio>

<audio controls>
  <source src="https://github.com/Wernerson/dl-project-2025/raw/refs/heads/master/samples/reference_1.wav" type="audio/mpeg">
</audio>

<audio controls>
  <source src="https://github.com/Wernerson/dl-project-2025/raw/refs/heads/master/samples/reference_1.mp4" type="audio/mpeg">
</audio>

Here are two references files:

<video src='samples/reference_1.mp4'/>

<video src='samples/reference_2.mp4'/>

# Experiments

These experiment files (`<name>.yaml`) correspond to the experiments in the paper as follows:

- `co-harch`: CoHierarchical
- `co-harch-useq`: CoHierarchical USeq
- `harch`: Hierarchical
- `harch-useq`: Hierarchical USeq
- `rand`: Random
- `rand-note`: Random Note
- `seq`: Sequential

Run experiments with for example: `experiment=co-hierarchical`.

# Masking Strategies

Masking strategies that can be configured further by parameters.

## `NoteMasking`

Masks `t` entire nodes randomly per step noise step.
Then unmasks one entire random note during the denoising step.

No parameters.

## `SequentialNoteMasking`

Unmasks all tokens from front-to-back/left-to-right.

No parameters.

## `ProbabilisticMasking`

Unmasks certain tokens with higher probability.

Parameters:

- `mask_token_id`: token ID to determine which tokens are not yet unmasked
- `samples_per_step`: determines how many samples are unmasked per step, `k` in paper and always equals 8
- `P_token`: 8 dimensional number array that determines token-probability, $P_{token}$ in paper
- `P_seq`: $n$-dimensional number array that determines sequential probabilities, , $P_{seq}$ in paper

# Repository Structure

Important files & directories:

- `cfg/`: config files for [hydra](https://hydra.cc/docs/intro/), we configure and instantiate models, classes, etc.
  - `dataset/*`: 
    - `lakh-MIDI-10k.yaml`: about 10k samples of LMD, used for training & evaluation
    - `lakh-MIDI-1k` is a smaller subset of LMD for development
  - `experiment/*`: see [Experiments](#experiments)
  - `run/*`: run configurations for different environments, set by `run=x`
    - `dev.yaml`: quick runs for development
    - `cluster.yaml`: runs on cluster
  - `config.yaml`: basic configuration parameters for all runs
- `src/`: Python files/code
  - `callbacks/generation.py`: creates samples each epoch & logs them to WandB
  - `dataset/miditok.py`: [Lightning DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) that downloads, extracts and splits LMD
  - `metrics/common.py`: extracts common [MusPy](https://muspy.readthedocs.io/en/stable/) metrics
  - `metrics/fmd.py`: extracts [Frechet Music Distance](https://github.com/jryban/frechet-music-distance)
  - `model/mask.py`: different masking strategies, see [Masking Strategies](#masking-strategies)
  - `model/musicbert_diffusion.py`: [Lightning Module](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) 
  - `net/musicbert.py`: the MusicBERT-derived model (neural net)
  - `config.py`: [hydra](https://hydra.cc/docs/intro/) `eval` resolver
  - `eval.py`: **main script for evaluation**
  - `evaluator.py`: copying reference files, generating samples & running metrics (in `metrics/*`)
  - `mask_vis.py`: script to visualise different masks 
  - `train.py`: **main script for training**
  - `utils.py`: MIDI to audio conversion
