# Agent-RL

Reinforcement learning agent models for computational modeling of behavioral data.

This project is heavily inspired by [Rasmus Bruckner](https://github.com/rasmusbruckner) and his [gaborbandit_analysis](https://github.com/rasmusbruckner/gaborbandit_analysis).

## Installation

```bash
git clone https://github.com/moltaire/agent-rl
cd agent-rl
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Quick Start

### Fitting behavioral data

Behavioral data should be a `pandas.DataFrame` with the following columns:

| Column | Description |
|--------|-------------|
| `block` | Block number (0-indexed) |
| `trial` | Trial number within block (0-indexed) |
| `s` | State observed by the agent |
| `a` | Action taken (0-indexed) |
| `r` | Reward received |

With this, you can fit a model to it like this:

```python
from agent_rl import (
    AgentVars, DualLearningRateAgent,
    EstimationVars, Estimation,
)

# Read data
data = pd.read_csv("data.csv")

# Create agent
agent_vars = AgentVars(alpha_pos=0.3, alpha_neg=0.2, beta=5.0)
agent = DualLearningRateAgent(agent_vars, n_options=2, n_states=4)

# Estimate parameters from behavioral data
estimator = Estimation(est_vars)
nll, bic, params, fitted_agent = estimator.estimate(data)
```

To simulate data (for recovery purposes, for example), you also need to set up a task:

```python
from agent_rl import (
    TaskVars, MultipleStatesTask,
    SingleLearningRateAgent,
    agent_task_interaction
)

# Set up task
task_vars = TaskVars(
    n_options=2,  # number of options shown in each trial
    n_blocks=4,  
    n_trials=100,
    states = {
        # Reward conditions for the different states
        #   p_r:       contains probability of first reward for choosing each option
        #   rewards:   contains reward outcomes (for p_r and (1 - p_r) respectively)
        #   a_correct: indicates correct action, used to compute accuracy. both actions can be correct. 
        # 
        # Example:
        # State 0 choosing 0 yields reward of 1 with p = 0.8, 0 otherwise
        # State 0 choosing 1 yields reward of 1 with p = 0.2, 0 otherwise
        0: {"p_r": [0.8, 0.2], "rewards": [1, 0], "a_correct": [0]},  
        1: {"p_r": [0.2, 0.8], "rewards": [1, 0], "a_correct": [1]},
        2: {"p_r": [0.5, 0.5], "rewards": [1, 0], "a_correct": [0, 1]},
        3: {"p_r": [0.25, 0.25], "rewards": [2, 0], "a_correct": [0, 1]},
    })
task = MultipleStatesTask(task_vars)

# Create agent
agent_vars = AgentVars(alpha=0.2, beta=7.0)
agent = SingleLearningRateAgent(agent_vars, n_options=2, n_states=4)

# Simulate data
sim_data = agent_task_interaction(task, agent)
```

## Agents

- **SingleLearningRateAgent** - Standard Rescorla-Wagner Q-learning with a single learning rate (α) for all prediction errors [1, 2]
- **DualLearningRateAgent** - Asymmetric learning with separate rates for positive (α+) and negative (α-) prediction errors [3, 4]

## Tasks

- **MultipleStateTask** - Multi-state bandit task with observable states and configurable reward probabilities [3, 5]
- **ReversalLearningTask** - Hidden-state reversal learning task with performance-based state switches [4]

## Examples

### Parameter recovery

In the [`parameter_recovery.ipynb`](notebooks/parameter_recovery.ipynb) notebook demonstrates a simple parameter recovery routine involving:

- Setup of a `ReversalLearningTask`
- Simulating data from `DualLearningRateAgent`s
- Parameter estimation for `DualLearningRateAgent`s
- Comparison of generating and recovered parameters

### Lefebvre et al. (2017) reproduction

In the [`lefebvre2017_analysis.ipynb`](notebooks/lefebvre2017_analysis.ipynb) notebook, I reproduce a main analysis of Lefebvre et al. [3], which includes:

- Setup of a `MultipleStatesTask`
- Fitting a `SingleLearningRateAgent` model
- and a `DualLearningRateAgent` model to each individual's data
- Model comparison
- Reproduction of Figure 3

Original data is available at [Figshare](https://figshare.com/articles/dataset/Behavioral_data_and_data_extraction_code/4265408/1?file=6949427). You can use this script to download the data automatically:

```{python}
uv run python scripts/download_lefebvre2017_data.py
```

## References

[1] Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement. In A. H. Black & W. F. Prokasy (Eds.), *Classical conditioning II: Current research and theory* (pp. 64-99). Appleton-Century-Crofts.

[2] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

[3] Lefebvre, G., Lebreton, M., Meyniel, F., Bourgeois-Gironde, S., & Palminteri, S. (2017). Behavioural and neural characterization of optimistic reinforcement learning. *Nature Human Behaviour*, 1(4), 1-9.

[4] Kahnt, T., Park, S. Q., Cohen, M. X., Beck, A., Heinz, A., & Wrase, J. (2009). Dorsal striatal–midbrain connectivity in humans predicts how reinforcements are used to guide decisions. *Journal of Cognitive Neuroscience*, 21(7), 1332-1345.

[5] Palminteri, S., Khamassi, M., Joffily, M., & Coricelli, G. (2015). Contextual modulation of value signals in reward and punishment learning. *Nature Communications*, 6, 8096.
