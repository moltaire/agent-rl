"""Agent-RL: Reinforcement learning agent models for behavioral data analysis."""

__version__ = "0.1.0"

from .agent import AgentVars, DualLearningRateAgent, SingleLearningRateAgent
from .task import TaskVars, MultipleStateTask, ReversalLearningTask
from .interaction import agent_task_interaction
from .estimation import EstimationVars, Estimation

__all__ = [
    "AgentVars",
    "DualLearningRateAgent",
    "SingleLearningRateAgent",
    "TaskVars",
    "MultipleStateTask",
    "ReversalLearningTask",
    "agent_task_interaction",
    "EstimationVars",
    "Estimation",
]
