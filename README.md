# Pendubot & Acrobot — Swing-up and Stabilization with SAC

**Leonardo Luigi Pepe — 2157562**  
Reinforcement Learning Project, University of Padua 2025/2026

A comparative study of classical control baselines and Soft Actor-Critic (SAC)-based controllers for the swing-up and stabilization of underactuated double pendulum systems: **Pendubot** and **Acrobot**.

---

## Quick Start

### Environment Setup

**macOS / Linux:**
```bash
bash setup.sh
conda activate double_pendulum
```

**Windows:**
```bat
setup.bat
conda activate double_pendulum
```

> On Windows, `setup.bat` requires Anaconda or Miniconda installed and added to PATH.  
> Download Miniconda: https://docs.conda.io/en/latest/miniconda.html

The setup script will:
- create a conda environment with Python 3.10,
- install `double_pendulum` from the local `src/` directory,
- install all required dependencies.

---

## Run the evaluation

After activating the environment, run:

```bash
python evaluate.py
```

This script evaluates the trained controllers and also runs the classical baseline for comparison.

`baseline.py` can also be executed separately if needed, but it is already called inside the evaluation workflow.

---

## Pre-trained models and results

The trained models are stored in:

- `pendubot_models/`
- `acrobot_models/`

Depending on the experiment, TensorBoard logs may also be available inside these folders.

The evaluation outputs are saved in the subfolders of:

- `results/`

Each result folder contains:
- a simulation video,
- a timeseries plot.

---

## Folder description

### `parameters/`
Contains the physical parameters of the two systems:
- Pendubot
- Acrobot

### `Training_pendubot/` and `Training_acrobot/`
These are the cleaned original folders used for model training.

If you want to verify the training pipeline, enter one of these folders and run:

```bash
python SAC_nomemodello_train.py
python Test_Agent.py
```

`Test_Agent.py` can also be executed while training is running.

> Training is **not recommended** unless strictly necessary, due to the very long execution times.

### `utility/`
Contains all auxiliary files required by `evaluate.py`, including:
- baseline controller files,
- global script configuration,
- metric evaluation utilities,
- custom controllers,
- RL environment configuration files,
- reward, termination, and reset functions.

---

## Main files

- `evaluate.py` — main script used to evaluate the trained agents
- `baseline.py` — standalone baseline execution
- `setup.sh` — installation script for a fresh machine
- `REPORT.pdf` — final project report
- `conda_environment.yml` — environment reference file

---

## Notes

- Run all commands from the **project root**.
- The project is designed to work after a fresh setup through `setup.sh`.
- The repository already includes the `src/` folder containing the `double_pendulum` package used by the project.