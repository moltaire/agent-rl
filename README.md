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

```python
from agent_rl import (
    AgentVars, DualLearningRateAgent,
    TaskVars, MultipleStateTask,
    agent_task_interaction,
    EstimationVars, Estimation,
)

# Create agent and simulate behavior
agent_vars = AgentVars(alpha_pos=0.3, alpha_neg=0.2, beta=5.0)
agent = DualLearningRateAgent(agent_vars, n_options=2, n_states=4)
data = agent_task_interaction(task, agent)

# Estimate parameters from behavioral data
estimator = Estimation(est_vars)
nll, bic, params, fitted_agent = estimator.estimate(data)
```

## Agents

- **SingleLearningRateAgent** - Standard Rescorla-Wagner Q-learning with a single learning rate (α) for all prediction errors [1, 2]
- **DualLearningRateAgent** - Asymmetric learning with separate rates for positive (α+) and negative (α-) prediction errors [3, 4]

## Tasks

- **MultipleStateTask** - Multi-state bandit task with observable states and configurable reward probabilities [3, 5]
- **ReversalLearningTask** - Hidden-state reversal learning task with performance-based state switches [4]

## Examples

- [Replication analysis of Lefebvre et al. (2017)](notebooks/lefebvre2017_analysis.ipynb) - Model comparison of single vs dual learning rate agents in a multi-state task.

## References

[1] Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement. In A. H. Black & W. F. Prokasy (Eds.), *Classical conditioning II: Current research and theory* (pp. 64-99). Appleton-Century-Crofts.

[2] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

[3] Lefebvre, G., Lebreton, M., Meyniel, F., Bourgeois-Gironde, S., & Palminteri, S. (2017). Behavioural and neural characterization of optimistic reinforcement learning. *Nature Human Behaviour*, 1(4), 1-9.

[4] Kahnt, T., Park, S. Q., Cohen, M. X., Beck, A., Heinz, A., & Wrase, J. (2009). Dorsal striatal–midbrain connectivity in humans predicts how reinforcements are used to guide decisions. *Journal of Cognitive Neuroscience*, 21(7), 1332-1345.

[5] Palminteri, S., Khamassi, M., Joffily, M., & Coricelli, G. (2015). Contextual modulation of value signals in reward and punishment learning. *Nature Communications*, 6, 8096.
