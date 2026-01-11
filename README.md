# MAML for BCI
This repository contains experiments investigating Model-Agnostic Meta-Learning (MAML) for rapid adaptation under neural and decoder drift, with applications to:
- Synthetic control tasks (continuous trajectory control with drift)
- EEG motor imagery under within-subject non-stationarity
The project accompanies a tutorial-style article and coursework for Reinforcement Learning for Bioengineers and focuses on reproducibility, modular code structure, and experimental design. The article can be found here:

# Project Overview
Neural signals are non-stationary. This project evaluates whether MAML can learn initial parameters that:
- adapt rapidly using only a few gradient steps
- outperform randomly initialised models under drift
Two settings are studied:
1. Synthetic control task
- Drift induced by decoder rotation and gain changes
- Continuous trajectory optimisation
2. EEG motor imagery
- Within-subject drift across recording runs
- Few-shot adaptation using support/query split

# Repository structure
```text
.
├── src/
│   ├── utils.py        # Seeding, statistics, helpers
│   ├── models.py       # Neural network models
│   ├── eeg_data.py     # EEG loading & task sampling
│   ├── train.py        # MAML training loops
│   └── evaluate.py     # Adaptation & evaluation routines
│
├── notebooks/
│   ├── demo_stationary.ipynb   # Synthetic drift experiments
│   └── demo_eeg.ipynb          # EEG within-subject drift
│
├── files/
│   └── SXXX/           # EEG dataset (not included)
│
├── requirements.txt
└── README.md
```


## Dataset
The EEG motor imagery dataset used in this project is **not included** in this repository due to licensing and data size considerations.

Please download the dataset from the original source:

https://doi.org/10.13026/C28G6P

After downloading, place the data in the following directory structure:
```text
files/
├── S001/
├── S002/
├── ...
```

# Running the Code
1. Install dependencies
   pip install -r requirements.txt
2. Synthetic experiments: This notebook trains MAML and baseline models, evaluates adaptation curves and generates figures used in the article
   notebooks/demo_stationary.ipynb
3. EEG experiments: place the data under files/
   Then open: notebooks/demo_eeg.ipynb

# Reproducibility
- All experiments are run with explicit random seeds
- Results are reported as mean ± SEM across seeds
- Models are evaluated on unseen tasks / subjects

# Academic Context
This project was developed as part of: Reinforcement Learning for Bioengineers (BIOE70077), Department of Bioengineering, Imperial College London. The work is intended as an educational tutorial.

# Use of Generative AI
Generative AI tools (ChatGPT 5.2) were used to:
- assist with code structuring
- debug implementation issues
- refine documentation
All experimental design, implementation decisions, and final results are the author’s own.
