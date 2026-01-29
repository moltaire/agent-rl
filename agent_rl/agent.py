"""Agent implementations for reinforcement learning models."""

import numpy as np


class AgentVars:
    """Container for agent parameters.

    This class allows flexible specification of agent parameters.

    Example:
        For a DualLearningRateAgent:
        >>> agent_vars = AgentVars(alpha_pos=0.3, alpha_neg=0.2, beta=5.0)

        For a SingleLearningRateAgent:
        >>> agent_vars = AgentVars(alpha=0.3, beta=5.0)
    """

    def __init__(self, **kwargs):
        """Initialize agent parameters.

        Args:
            **kwargs: Arbitrary keyword arguments for agent parameters.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, **kwargs):
        """Update agent parameters.

        Args:
            **kwargs: Parameters to update.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)


class DualLearningRateAgent:
    """Dual learning rate reinforcement learning agent.

    This agent implements asymmetric learning rates for positive and negative
    prediction errors (Lefebvre et al. 2017) or rewards (Kahnt et al. 2008).

    The agent uses a standard delta rule for learning:
        Q(s,a) += α * (r - Q(s,a))

    where α depends on the sign of either:
    - The prediction error δ = r - Q(s,a) (variant='delta', Lefebvre et al. 2017)
    - The reward r (variant='r', Kahnt et al. 2008)

    Choice is determined by softmax with inverse temperature β:
        P(a|s) = exp(β * Q(s,a)) / Σ exp(β * Q(s,a'))

    Attributes:
        agent_vars (AgentVars): Agent parameters (alpha_pos, alpha_neg, beta).
        options (range): Available action options.
        variant (str): Model variant ('delta' or 'r').
        Q_t (np.ndarray): Current Q-values [n_states × n_options].
        a_t (int): Current action.
        s_t (int): Current state.
        p_a_t (np.ndarray): Current action probabilities.
    """

    def __init__(self, agent_vars, n_options, n_states=1, variant='delta'):
        """Initialize the dual learning rate agent.

        Args:
            agent_vars (AgentVars): Agent parameters. Must have alpha_pos, alpha_neg, beta.
            n_options (int): Number of actions available.
            n_states (int, optional): Number of states. Defaults to 1.
            variant (str, optional): Learning rate variant. Either 'delta' (Lefebvre et al. 2017)
                where learning rate depends on prediction error sign, or 'r' (Kahnt et al. 2008)
                where it depends on reward sign. Defaults to 'delta'.
        """
        self.check_agent_vars(agent_vars)
        self.agent_vars = agent_vars
        self.options = range(n_options)
        self.variant = variant

        # Initial Q-values
        if not hasattr(agent_vars, "Q_init"):
            self.agent_vars.Q_init = np.zeros((n_states, n_options))

        # Allow variant specification via agent_vars
        if hasattr(agent_vars, "variant"):
            if (self.variant is not None) and (self.variant != agent_vars.variant):
                print(
                    f"Overriding variant: {self.variant} -> {agent_vars.variant}"
                )
            self.variant = agent_vars.variant

        self.Q_t = self.agent_vars.Q_init.copy()
        self.a_t = None
        self.s_t = 0
        self.p_a_t = None

    def check_agent_vars(self, agent_vars):
        """Validate that required parameters are present."""
        for var in ["alpha_pos", "alpha_neg", "beta"]:
            if not hasattr(agent_vars, var):
                raise ValueError(f"agent_vars is missing `{var}` attribute.")

    def __repr__(self):
        return (
            f"DualLearningRateAgent(variant='{self.variant}', "
            f"alpha_pos={self.agent_vars.alpha_pos:.3f}, "
            f"alpha_neg={self.agent_vars.alpha_neg:.3f}, "
            f"beta={self.agent_vars.beta:.3f})"
        )

    def softmax(self, Q_s_t):
        """Compute softmax choice probabilities.

        Args:
            Q_s_t (np.ndarray): Q-values for current state.

        Returns:
            np.ndarray: Action probabilities.
        """
        exp_values = np.exp(Q_s_t * self.agent_vars.beta)
        return exp_values / np.sum(exp_values)

    def observe_state(self, task):
        """Observe the current task state.

        Args:
            task: Task object with show_state() method.
        """
        self.s_t = task.show_state()

    def decide(self):
        """Make a choice based on current Q-values using softmax."""
        self.p_a_t = self.softmax(self.Q_t[self.s_t, :])
        self.a_t = np.random.choice(self.options, p=self.p_a_t)

    def learn(self, r_t):
        """Update Q-values based on received reward.

        Args:
            r_t (float): Received reward.
        """
        delta = r_t - self.Q_t[self.s_t, self.a_t]

        # Determine learning rate based on variant
        if self.variant == "r":
            reference_var = r_t
        elif self.variant == "delta":
            reference_var = delta
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        # Select appropriate learning rate
        if reference_var > 0:
            alpha = self.agent_vars.alpha_pos
        else:
            alpha = self.agent_vars.alpha_neg

        # Update Q-value
        self.Q_t[self.s_t, self.a_t] += alpha * delta

    def reset(self):
        """Reset Q-values to initial values."""
        self.Q_t = self.agent_vars.Q_init.copy()
        self.a_t = None
        self.s_t = 0
        self.p_a_t = None


class SingleLearningRateAgent:
    """Single learning rate reinforcement learning agent.

    This agent implements the classic Rescorla-Wagner delta rule with a single
    learning rate for all prediction errors.

    Learning rule:
        Q(s,a) += α * (r - Q(s,a))

    Choice rule (softmax):
        P(a|s) = exp(β * Q(s,a)) / Σ exp(β * Q(s,a'))

    Attributes:
        agent_vars (AgentVars): Agent parameters (alpha, beta).
        options (range): Available action options.
        Q_t (np.ndarray): Current Q-values [n_states × n_options].
        a_t (int): Current action.
        s_t (int): Current state.
        p_a_t (np.ndarray): Current action probabilities.
    """

    def __init__(self, agent_vars, n_options, n_states=1):
        """Initialize the single learning rate agent.

        Args:
            agent_vars (AgentVars): Agent parameters. Must have alpha and beta.
            n_options (int): Number of actions available.
            n_states (int, optional): Number of states. Defaults to 1.
        """
        self.check_agent_vars(agent_vars)
        self.agent_vars = agent_vars
        self.options = range(n_options)

        # Initial Q-values
        if not hasattr(agent_vars, "Q_init"):
            self.agent_vars.Q_init = np.zeros((n_states, n_options))

        self.Q_t = self.agent_vars.Q_init.copy()
        self.a_t = None
        self.s_t = 0
        self.p_a_t = None

    def check_agent_vars(self, agent_vars):
        """Validate that required parameters are present."""
        for var in ["alpha", "beta"]:
            if not hasattr(agent_vars, var):
                raise ValueError(f"agent_vars is missing `{var}` attribute.")

    def __repr__(self):
        return (
            f"SingleLearningRateAgent("
            f"alpha={self.agent_vars.alpha:.3f}, "
            f"beta={self.agent_vars.beta:.3f})"
        )

    def softmax(self, Q_s_t):
        """Compute softmax choice probabilities.

        Args:
            Q_s_t (np.ndarray): Q-values for current state.

        Returns:
            np.ndarray: Action probabilities.
        """
        exp_values = np.exp(Q_s_t * self.agent_vars.beta)
        return exp_values / np.sum(exp_values)

    def observe_state(self, task):
        """Observe the current task state.

        Args:
            task: Task object with show_state() method.
        """
        self.s_t = task.show_state()

    def decide(self):
        """Make a choice based on current Q-values using softmax."""
        self.p_a_t = self.softmax(self.Q_t[self.s_t, :])
        self.a_t = np.random.choice(self.options, p=self.p_a_t)

    def learn(self, r_t):
        """Update Q-values based on received reward.

        Args:
            r_t (float): Received reward.
        """
        delta = r_t - self.Q_t[self.s_t, self.a_t]
        self.Q_t[self.s_t, self.a_t] += self.agent_vars.alpha * delta

    def reset(self):
        """Reset Q-values to initial values."""
        self.Q_t = self.agent_vars.Q_init.copy()
        self.a_t = None
        self.s_t = 0
        self.p_a_t = None
