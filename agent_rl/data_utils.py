"""Utilities for loading experimental data."""

import re
import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_lefebvre_subject(matfile, experiment):
    """Load and process a single subject's data from Lefebvre et al. (2017).

    Args:
        matfile (str): Path to .mat file.
        experiment (int): Experiment number (1 or 2). Data format differs between experiments.

    Returns:
        pd.DataFrame: Formatted data with columns: subject, block, trial, s, a, r.

    Notes:
        - Experiment 1: 52 subjects, 96 trials each
        - Experiment 2: 38 subjects, 96 trials each
        - Both experiments use the same 4-state task structure
        - States: 0=75/25, 1=25/25, 2=25/75, 3=75/75
        - Actions: 0 or 1 (left or right)
        - Rewards in Exp1: {0, 0.5}, Exp2: {-0.5, 0.5}
    """
    # Extract subject ID from filename
    match = re.search(r"_(\d+)\.mat", matfile)
    if match:
        subject = int(match.group(1))
    else:
        raise ValueError(f"Cannot extract subject ID from filename: {matfile}")

    # Column names differ between experiments
    if experiment == 1:
        columns = ["_", "trial", "s", "_", "_", "_", "a", "r", "_"]
    elif experiment == 2:
        columns = ["trial", "s", "_", "_", "_", "a", "r"]
    else:
        raise ValueError(f"experiment must be 1 or 2, got {experiment}")

    # Load .mat file
    mat = loadmat(matfile)
    df = pd.DataFrame(mat["data"], columns=columns)

    # Process variables
    df["subject"] = subject
    df["block"] = 0
    df["trial"] = (df["trial"] - 1).astype(np.int32)  # 0-indexed
    df["a"] = ((df["a"] / 2 + 0.5).astype(np.int32))  # [-1, 1] -> [0, 1]
    df["s"] = (df["s"] - 1).astype(np.int32)  # [1,2,3,4] -> [0,1,2,3]

    # Rewards differ by experiment
    if experiment == 1:
        df["r"] = df["r"] / 2  # [0, 1] -> [0, 0.5]

    return df[["subject", "block", "trial", "s", "a", "r"]]


def load_lefebvre_experiment(data_dir, experiment):
    """Load all subjects from a Lefebvre et al. (2017) experiment.

    Args:
        data_dir (str): Directory containing .mat files (e.g., 'data/lefebvre2017/data_exp1').
        experiment (int): Experiment number (1 or 2).

    Returns:
        pd.DataFrame: Combined data for all subjects.
    """
    from pathlib import Path
    import glob

    # Find all .mat files
    pattern = str(Path(data_dir) / f"exp{experiment}_*.mat")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No .mat files found matching {pattern}")

    # Load and concatenate
    dfs = []
    for f in files:
        try:
            df = load_lefebvre_subject(f, experiment)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    if not dfs:
        raise ValueError(f"No data loaded from {data_dir}")

    return pd.concat(dfs, ignore_index=True)


def get_lefebvre_task_vars(experiment=2):
    """Get task configuration for Lefebvre et al. (2017) experiments.

    Args:
        experiment (int, optional): Experiment number (1 or 2). Defaults to 2.

    Returns:
        TaskVars: Task configuration with states and state_sequence.

    Notes:
        Both experiments used the same task structure:
        - 4 states (conditions): 75/25, 25/25, 25/75, 75/75
        - 2 options per state
        - 96 trials (24 per state, randomized)
        - Rewards: Exp1 {0, 0.5}, Exp2 {-0.5, 0.5}
    """
    from .task import TaskVars

    # Define states (same for both experiments)
    if experiment == 1:
        rewards = [0.5, 0]
    elif experiment == 2:
        rewards = [0.5, -0.5]
    else:
        raise ValueError(f"experiment must be 1 or 2, got {experiment}")

    states = {
        0: {"p_r": [0.75, 0.25], "rewards": rewards, "a_correct": [0]},
        1: {"p_r": [0.25, 0.25], "rewards": rewards, "a_correct": [0, 1]},
        2: {"p_r": [0.25, 0.75], "rewards": rewards, "a_correct": [1]},
        3: {"p_r": [0.75, 0.75], "rewards": rewards, "a_correct": [0, 1]},
    }

    # Create random state sequence (used for simulation, not estimation)
    # 24 trials per state, randomized
    n_repeats = 24
    n_states = len(states)
    state_sequence = np.repeat(np.arange(n_states), n_repeats)
    np.random.shuffle(state_sequence)
    state_sequence = state_sequence.reshape((1, -1))

    task_vars = TaskVars(n_options=2, n_blocks=1, n_trials=96)
    task_vars.states = states
    task_vars.state_sequence = state_sequence
    task_vars.n_states = len(states)

    return task_vars
