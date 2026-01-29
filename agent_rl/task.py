"""Task implementations for reinforcement learning experiments."""

import numpy as np


class TaskVars:
    """Container for task parameters.

    This class allows flexible specification of task parameters.

    Args:
        n_trials (int, optional): Number of trials per block.
        n_blocks (int, optional): Number of blocks.
        n_options (int, optional): Number of action options.
        n_states (int, optional): Number of different states. Defaults to 1.
        **kwargs: Additional task-specific parameters (e.g., states, rewards).

    Example:
        >>> task_vars = TaskVars(n_trials=96, n_blocks=1, n_options=2)
        >>> task_vars.states = {0: {"p_r": [0.75, 0.25], "rewards": [1, -1]}}
    """

    def __init__(
        self, n_trials=None, n_blocks=None, n_options=None, n_states=1, **kwargs
    ):
        self.n_trials = n_trials
        self.n_blocks = n_blocks
        self.n_options = n_options
        self.n_states = n_states

        for key, value in kwargs.items():
            setattr(self, key, value)


class MultipleStateTask:
    """Multiple-state instrumental learning task.

    This task is modeled after Lefebvre et al. (2017) and Palminteri et al. (2015).
    Different states (conditions) are presented across trials, each with its own
    set of options and reward probabilities. States vary between trials according
    to a predefined sequence.

    For example, in the Lefebvre et al. study, there are 4 conditions with 2 options
    each, resulting in 8 different symbols whose values participants must learn.

    Attributes:
        kind (str): Task type identifier.
        task_vars (TaskVars): Task parameters including states and state_sequence.
        trial (int): Current trial within block.
        block (int): Current block number.
        state_t (int): Current state/condition.
        r_t (float): Current reward.
        correct_t (bool): Whether current action was correct.
    """

    def __init__(self, task_vars):
        """Initialize multiple-state task.

        Args:
            task_vars (TaskVars): Task parameters. Must include:
                - states (dict): State definitions with p_r, rewards, a_correct
                - state_sequence (np.ndarray): [n_blocks × n_trials] state sequence
        """
        self.kind = "multiple-state"
        self.check_task_vars(task_vars)
        self.task_vars = task_vars
        self.trial = 0
        self.block = 0

        # Derive dimensions from state_sequence
        (
            self.task_vars.n_blocks,
            self.task_vars.n_trials,
        ) = self.task_vars.state_sequence.shape
        self.task_vars.n_states = len(task_vars.states)

        self.state_t = None
        self.r_t = None
        self.correct_t = None

    def check_task_vars(self, task_vars):
        """Validate required task parameters."""
        if not hasattr(task_vars, "states"):
            raise ValueError("task_vars must have 'states' attribute")
        if not hasattr(task_vars, "state_sequence"):
            raise ValueError("task_vars must have 'state_sequence' attribute")
        return True

    def __repr__(self):
        return f"MultipleStateTask(n_states={self.task_vars.n_states})"

    def reset(self):
        """Reset task to initial state."""
        self.trial = 0
        self.block = 0
        self.state_t = None
        self.r_t = None
        self.correct_t = None

    def prepare_trial(self, verbose=False):
        """Prepare next trial by selecting state from sequence.

        Advances trial and block counters.
        """
        self.state_t = self.task_vars.state_sequence[self.block, self.trial]

        self.trial += 1

        # Check if end of block reached
        if self.trial >= self.task_vars.n_trials:
            self.block += 1
            self.trial = 0

    def show_state(self):
        """Show current state to agent.

        In this task, states are observable (different symbols per condition).

        Returns:
            int: Current state.
        """
        return self.state_t

    def get_p_r_a(self, a_t):
        """Get reward probability for given action.

        Args:
            a_t (int): Action taken.

        Returns:
            float: Probability of reward.
        """
        return self.task_vars.states[self.state_t]["p_r"][a_t]

    def sample_reward(self, a_t):
        """Sample reward for given action.

        Args:
            a_t (int): Action taken.
        """
        p_r = self.get_p_r_a(a_t)
        self.r_t = np.random.choice(
            self.task_vars.states[self.state_t]["rewards"], p=[p_r, 1 - p_r]
        )

        self.correct_t = a_t in self.task_vars.states[self.state_t]["a_correct"]


class ReversalLearningTask:
    """Reversal learning task.

    This task is modeled after Kahnt et al. (2008). Different hidden states (rules)
    determine reward probabilities, and the task reverses between states based on
    performance criteria.

    Reversal occurs when:
    - Minimum trials completed AND minimum accuracy reached, OR
    - Maximum trials reached (forced reversal)

    Attributes:
        kind (str): Task type identifier.
        task_vars (TaskVars): Task parameters.
        state_t (int): Current (hidden) state.
        n_trials_current_state (int): Trials in current state.
        n_correct_current_state (int): Correct responses in current state.
        p_correct_current_state (float): Accuracy in current state.
        r_t (float): Current reward.
        correct_t (bool): Whether current action was correct.
    """

    def __init__(self, task_vars):
        """Initialize reversal learning task.

        Args:
            task_vars (TaskVars): Task parameters. Must include:
                - states (dict): State definitions with p_r, rewards, a_correct
                - n_trials_reversal_min (int): Minimum trials before reversal
                - n_trials_reversal_max (int): Maximum trials (forced reversal)
                - p_correct_reversal_min (float): Accuracy threshold for reversal
        """
        self.kind = "reversal-learning"
        self.check_task_vars(task_vars)
        self.task_vars = task_vars

        # Initialize with random state
        self.state_t = np.random.choice(list(task_vars.states.keys()))

        # Reversal tracking variables
        self.n_trials_current_state = 0
        self.n_correct_current_state = 0
        self.p_correct_current_state = 0
        self.correct_t = None
        self.r_t = None

    def check_task_vars(self, task_vars):
        """Validate required task parameters."""
        required = ["states", "n_trials_reversal_min", "n_trials_reversal_max",
                    "p_correct_reversal_min"]
        for attr in required:
            if not hasattr(task_vars, attr):
                raise ValueError(f"task_vars must have '{attr}' attribute")
        return True

    def __repr__(self):
        return f"ReversalLearningTask(n_states={len(self.task_vars.states)})"

    def reset(self):
        """Reset task to initial state with random rule."""
        self.state_t = np.random.choice(list(self.task_vars.states.keys()))
        self.n_trials_current_state = 0
        self.n_correct_current_state = 0
        self.p_correct_current_state = 0
        self.correct_t = None
        self.r_t = None

    def prepare_trial(self, verbose=False):
        """Prepare next trial and check for reversals.

        Reversal occurs if:
        - n_trials >= min AND accuracy >= threshold, OR
        - n_trials == max (forced)
        """
        reversal_t = False

        # Check reversal conditions
        if (self.n_trials_current_state > self.task_vars.n_trials_reversal_min) and (
            self.p_correct_current_state >= self.task_vars.p_correct_reversal_min
        ):
            reversal_t = True
        elif self.n_trials_current_state == self.task_vars.n_trials_reversal_max:
            reversal_t = True

        # Execute reversal
        if reversal_t:
            current_state = self.state_t
            other_states = [
                state
                for state in list(self.task_vars.states.keys())
                if state != current_state
            ]
            new_state = np.random.choice(other_states)

            if verbose:
                print(f"\n*** Reversal: State {current_state} -> State {new_state} ***\n")

            self.state_t = new_state
            self.n_trials_current_state = 0
            self.n_correct_current_state = 0
            self.p_correct_current_state = 0

    def show_state(self):
        """Show state to agent.

        In reversal learning, states are hidden. All trials look identical,
        so this always returns 0 regardless of true state.

        Returns:
            int: Always 0 (hidden state).
        """
        return 0

    def get_p_r_a(self, a_t):
        """Get reward probability for given action.

        Args:
            a_t (int): Action taken.

        Returns:
            float: Probability of reward.
        """
        return self.task_vars.states[self.state_t]["p_r"][a_t]

    def sample_reward(self, a_t):
        """Sample reward for given action and update reversal tracking.

        Args:
            a_t (int): Action taken.
        """
        p_r = self.get_p_r_a(a_t)
        self.r_t = np.random.choice(
            self.task_vars.states[self.state_t]["rewards"], p=[p_r, 1 - p_r]
        )

        # Update reversal tracking
        self.n_trials_current_state += 1
        self.correct_t = a_t in self.task_vars.states[self.state_t]["a_correct"]
        self.n_correct_current_state += int(self.correct_t)
        self.p_correct_current_state = (
            self.n_correct_current_state / self.n_trials_current_state
        )
