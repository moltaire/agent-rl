"""Parameter estimation via maximum likelihood."""

import numpy as np
from scipy.optimize import minimize


class EstimationVars:
    """Container for estimation configuration.

    Attributes:
        task_vars: Task parameters (n_trials, n_blocks, n_options, n_states).
        agent_class: Agent class to fit (e.g., DualLearningRateAgent).
        parameters (list): Parameter names to estimate.
        bounds (dict): Parameter bounds as {param: (min, max)} tuples.
        n_sp (int): Number of random starting points for optimization.
        fixed_sp (dict, optional): Fixed starting points {param: value}.
        rand_sp (bool): Use random (True) or fixed (False) starting points.
        reset_q_each_block (bool): Whether to reset Q-values between blocks.
    """

    def __init__(
        self,
        task_vars,
        agent_class,
        parameters,
        bounds,
        fixed_sp=None,
        n_sp=1,
        rand_sp=True,
        reset_q_each_block=False,
        variant='delta',
    ):
        """Initialize estimation configuration.

        Args:
            task_vars: Task parameters object.
            agent_class: Agent class to estimate (e.g., DualLearningRateAgent).
            parameters (list): Names of parameters to estimate.
            bounds (dict): Parameter bounds {name: (min, max)}.
            fixed_sp (dict, optional): Fixed starting points {name: value}.
            n_sp (int, optional): Number of random starting points. Defaults to 1.
            rand_sp (bool, optional): Use random starting points. Defaults to True.
            reset_q_each_block (bool, optional): Reset Q-values each block. Defaults to False.
            variant (str, optional): Agent variant (for DualLearningRateAgent). Defaults to 'delta'.
        """
        self.n_trials = task_vars.n_trials
        self.n_blocks = task_vars.n_blocks
        self.n_options = task_vars.n_options
        self.n_states = task_vars.n_states
        self.agent_class = agent_class
        self.parameters = parameters
        self.n_params = len(parameters)
        self.bounds = bounds
        self.fixed_sp = fixed_sp
        self.n_sp = n_sp
        self.rand_sp = rand_sp
        self.reset_q_each_block = reset_q_each_block
        self.variant = variant


class Estimation:
    """Maximum likelihood parameter estimation.

    This class implements parameter fitting via negative log-likelihood
    minimization using scipy.optimize.minimize with L-BFGS-B algorithm.

    The estimation process:
    1. Initializes agent with candidate parameters
    2. Simulates agent choices given observed states and outcomes
    3. Computes log-likelihood of observed choices
    4. Minimizes negative log-likelihood to find best parameters

    Attributes:
        est_vars (EstimationVars): Estimation configuration.
    """

    def __init__(self, est_vars):
        """Initialize estimator.

        Args:
            est_vars (EstimationVars): Estimation configuration.
        """
        self.est_vars = est_vars

    def nll(self, x, data, agent_vars, debug=False, return_agent=False):
        """Compute negative log-likelihood of observed choices.

        Given parameter values, this simulates the agent's choice probabilities
        for each trial and computes the negative log-likelihood of the observed
        choices.

        Args:
            x (np.ndarray): Parameter values in order of est_vars.parameters.
            data (pd.DataFrame): Observed data with columns: block, s, a, r.
            agent_vars: Agent variables object (updated with current parameters).
            debug (bool or int, optional): Verbosity level. Defaults to False.
            return_agent (bool, optional): Return fitted agent object. Defaults to False.

        Returns:
            float or tuple: Negative log-likelihood, or (NLL, agent) if return_agent=True.
        """
        # Map parameter array to named parameters
        parameters = {
            self.est_vars.parameters[i]: x[i] for i in range(len(x))
        }

        # Update agent parameters
        agent_vars.update(**parameters)
        agent_vars.reset_q_each_block = self.est_vars.reset_q_each_block

        # Set variant in agent_vars for compatibility
        if hasattr(self.est_vars, 'variant'):
            agent_vars.variant = self.est_vars.variant

        # Initialize agent
        # Try to pass variant if it's a DualLearningRateAgent
        try:
            agent = self.est_vars.agent_class(
                agent_vars=agent_vars,
                n_options=self.est_vars.n_options,
                n_states=self.est_vars.n_states,
                variant=self.est_vars.variant,
            )
        except TypeError:
            # For agents that don't have variant parameter (e.g., SingleLearningRateAgent)
            agent = self.est_vars.agent_class(
                agent_vars=agent_vars,
                n_options=self.est_vars.n_options,
                n_states=self.est_vars.n_states,
            )

        # Initialize likelihood storage
        nll_a = np.full(
            [self.est_vars.n_trials, self.est_vars.n_blocks], np.nan
        )

        # Process each block
        blocks = data["block"].unique()
        for b, block in enumerate(blocks):
            # Reset Q-values if specified
            if agent_vars.reset_q_each_block:
                agent.reset()
                if debug > 0:
                    print(f"  Block {block}: Reset Q-values to {agent.Q_t}")

            # Extract block data
            block_data = data[data["block"] == block]
            a = block_data["a"].values
            r = block_data["r"].values
            s = block_data["s"].values

            if debug > 0:
                print(f"  Block {block}: Initial Q = {agent.Q_t}")

            # Process each trial
            for t in range(self.est_vars.n_trials):
                # Skip if no action (missing data)
                if np.isnan(a[t]):
                    continue

                # Set observed state
                agent.s_t = int(s[t])

                # Compute choice probabilities
                if debug > 1:
                    print(f"    Trial {t}: s={s[t]}, a={a[t]}, r={r[t]}")
                    print(f"      Pre-decision Q = {agent.Q_t}")

                agent.decide()

                # Override action with observed action
                agent.a_t = int(a[t])

                # Record log-likelihood
                nll_a[t, b] = np.log(agent.p_a_t[int(a[t])])

                # Update Q-values
                agent.learn(r[t])

                if debug > 1:
                    print(f"      Post-learning Q = {agent.Q_t}")

            if debug > 0:
                print(f"  Block {block}: Final Q = {agent.Q_t}")

        # Sum negative log-likelihoods
        nll = -1 * np.nansum(nll_a)

        if return_agent:
            return (nll, agent)
        else:
            return nll

    def estimate(self, data, agent_vars=None, seed=None, debug=False):
        """Estimate parameters via maximum likelihood.

        Uses multiple random starting points and selects the best fit
        (lowest negative log-likelihood).

        Args:
            data (pd.DataFrame): Observed data with columns: block, s, a, r.
            agent_vars: Agent variables (optional, created if None).
            seed (int, optional): Random seed for reproducibility.
            debug (bool or int, optional): Verbosity level. Defaults to False.

        Returns:
            tuple: (nll, bic, parameters, agent)
                - nll (float): Negative log-likelihood at minimum
                - bic (float): Bayesian Information Criterion
                - parameters (list): Fitted parameter values
                - agent: Fitted agent object
        """
        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Initialize agent vars if needed
        if agent_vars is None:
            from .agent import AgentVars
            agent_vars = AgentVars()

        # Ensure Q_init is set
        if not hasattr(agent_vars, 'Q_init'):
            agent_vars.Q_init = np.zeros(
                (self.est_vars.n_states, self.est_vars.n_options)
            )

        # Track best result across starting points
        min_nll = np.inf
        min_x = None

        # Prepare bounds
        bounds = [
            self.est_vars.bounds[param]
            for param in self.est_vars.parameters
        ]

        # Try multiple starting points
        for r in range(self.est_vars.n_sp):
            # Set starting point
            if self.est_vars.rand_sp:
                x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
            else:
                x0 = [
                    self.est_vars.fixed_sp[param]
                    for param in self.est_vars.parameters
                ]

            if debug:
                print(f"\nStarting point {r+1}/{self.est_vars.n_sp}: {x0}")

            # Run optimization
            result = minimize(
                self.nll,
                x0,
                args=(data, agent_vars, debug),
                method="L-BFGS-B",
                bounds=bounds,
            )

            # Check if this is the best result
            if result.fun < min_nll:
                min_nll = result.fun
                min_x = result.x

                if debug:
                    print(f"  New best: NLL = {min_nll:.3f}, params = {min_x}")

        # Get fitted agent with best parameters
        _, agent = self.nll(
            min_x, data=data, agent_vars=agent_vars,
            debug=False, return_agent=True
        )

        # Compute BIC
        bic = self.compute_bic(min_nll, self.est_vars.n_params)

        return min_nll, bic, min_x.tolist(), agent

    def compute_bic(self, nll, n_params):
        """Compute Bayesian Information Criterion.

        BIC = -2*LL + k*ln(N)
        where LL is log-likelihood, k is number of parameters, N is sample size.

        Args:
            nll (float): Negative log-likelihood.
            n_params (int): Number of free parameters.

        Returns:
            float: BIC value.
        """
        N = self.est_vars.n_trials * self.est_vars.n_blocks
        LL = -nll
        BIC = -2 * LL + n_params * np.log(N)
        return BIC
