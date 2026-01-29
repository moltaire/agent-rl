"""Agent-task interaction simulation."""

import numpy as np
import pandas as pd


def agent_task_interaction(task, agent, verbose=False):
    """Simulate interaction between task and agent.

    This function runs a complete task session where an agent interacts with
    a task environment, making choices and learning from outcomes.

    The interaction follows this sequence per trial:
    1. Task prepares trial (sets state, checks for reversals)
    2. Agent observes state
    3. Agent makes choice (action)
    4. Task samples reward based on action
    5. Agent learns from reward

    Args:
        task: Task object (MultipleStateTask or ReversalLearningTask).
        agent: Agent object (DualLearningRateAgent or SingleLearningRateAgent).
        verbose (bool or int, optional): Verbosity level. False/0 for none,
            1 for blocks, 2+ for trials. Defaults to False.

    Returns:
        pd.DataFrame: Trial-by-trial data with columns:
            Task variables:
                - trial: Overall trial number
                - block: Block number
                - state: True task state/condition
                - p_r_i: Reward probability for each option i
                - ev_i: Expected value for each option i
                - r: Reward received
                - corr: Whether action was correct
            Agent variables:
                - a: Action chosen
                - s: State observed by agent
                - p_a_i: Choice probability for each option i
                - Q_i: Q-value (pre-update) for each option i
                - ll: Log-likelihood of choice
    """
    # Extract task parameters
    n_trials = task.task_vars.n_trials
    n_blocks = task.task_vars.n_blocks
    n_options = task.task_vars.n_options
    n_states = task.task_vars.n_states

    T = n_trials * n_blocks

    if verbose:
        print(f"Simulating {T} trials ({n_blocks} blocks × {n_trials} trials)...")

    # Reset counters
    task.block = 0
    task.trial = 0

    # Initialize arrays for data storage
    # Task variables
    trial = np.full(T, np.nan)
    block = np.full(T, np.nan)
    state = np.full(T, np.nan)
    p_r = np.full((T, n_options), np.nan)
    ev = np.full((T, n_options), np.nan)
    r = np.full(T, np.nan)

    # Agent variables
    a = np.full(T, np.nan)
    s = np.full(T, np.nan)
    corr = np.full(T, np.nan)
    p_a = np.full((T, n_options), np.nan)
    Q = np.full((T, n_options), np.nan)
    ll = np.full(T, np.nan)

    t = 0

    # Main simulation loop
    for b in range(n_blocks):
        if verbose:
            print(f"Block {b}")

        # Check if agent should reset Q-values between blocks
        if hasattr(agent.agent_vars, 'reset_q_each_block') and agent.agent_vars.reset_q_each_block:
            agent.reset()
            if verbose:
                print(f"  Reset Q-values: {agent.Q_t}")

        # Reset task (for reversal learning, resets counters)
        if hasattr(task, 'reset'):
            task.reset()

        for t_b in range(n_trials):
            if verbose > 1:
                print(f"  Trial {t_b}")

            # --- Trial sequence ---

            # 1. Prepare trial
            task.prepare_trial()

            # 2. Agent observes state
            agent.observe_state(task)
            if verbose > 1:
                print(f"    State: {task.state_t}, Observed: {agent.s_t}")

            # 3. Agent makes choice
            agent.decide()
            if verbose > 1:
                print(f"    p_a: {agent.p_a_t}")
                print(f"    Action: {agent.a_t}")

            # 4. Task returns reward
            task.sample_reward(agent.a_t)

            # 5. Agent learns (save pre-learning Q-values first)
            Q_t = agent.Q_t.copy()
            if verbose > 1:
                print(f"    Reward: {task.r_t}")
                print(f"    Pre-learning Q: {agent.Q_t[agent.s_t, :]}")
            agent.learn(task.r_t)
            if verbose > 1:
                print(f"    Post-learning Q: {agent.Q_t[agent.s_t, :]}")

            # --- Record data ---

            # Task variables
            trial[t] = t
            block[t] = b
            state[t] = task.state_t
            r[t] = task.r_t

            for i in range(n_options):
                p_r[t, i] = task.task_vars.states[task.state_t]["p_r"][i]
                # Compute expected value
                ev[t, i] = (
                    task.task_vars.states[task.state_t]["p_r"][i]
                    * task.task_vars.states[task.state_t]["rewards"][0]
                    + (1 - task.task_vars.states[task.state_t]["p_r"][i])
                    * task.task_vars.states[task.state_t]["rewards"][1]
                )

            # Agent variables
            a[t] = agent.a_t
            s[t] = agent.s_t
            corr[t] = task.correct_t

            for i in range(n_options):
                p_a[t, i] = agent.p_a_t[i]
                # Record pre-learning Q-values (used for choice)
                Q[t, i] = Q_t[agent.s_t, i]

            ll[t] = np.log(agent.p_a_t[int(a[t])])

            t += 1

    # Assemble DataFrame
    df = pd.DataFrame({
        "trial": trial,
        "block": block,
        "state": state,
        "r": r,
        "a": a,
        "s": s,
        "corr": corr,
        "ll": ll,
    })

    # Add option-specific columns
    for i in range(n_options):
        df[f"p_r_{i}"] = p_r[:, i]
        df[f"ev_{i}"] = ev[:, i]
        df[f"p_a_{i}"] = p_a[:, i]
        df[f"Q_{i}"] = Q[:, i]

    return df
