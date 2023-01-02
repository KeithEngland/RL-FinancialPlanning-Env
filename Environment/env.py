The environment consists of a portfolio of buyable securities with a certain amount of money invested in each. The agent
can make trades by adjusting the proportions of money invested in each security. The environment also has a client with
an age and a target spending rate. The goal of the agent is to maximize the value of the portfolio while ensuring that
the client does not run out of money before reaching a certain age.

The observation space consists of the client's age, the value of the portfolio, the target spending rate, and the
proportions of the portfolio invested in each security. The action space consists of the proportions of the portfolio
to invest in each security.

Attributes
----------
count_buyable_securities : int
    The number of buyable securities in the portfolio.
observation_space : gym.spaces.Box
    The observation space for the environment.
action_space : gym.spaces.Box
    The action space for the environment.
stepsPerYear : int
    The number of steps per year in the environment.
min_start_age : float
    The minimum starting age for the client.
max_start_age : float
    The maximum starting age for the client.
age_step_size : float
    The size of each age step in the environment.
start_age : float
    The starting age for the client.
secExpRet : numpy.ndarray
    The expected return for each security.
secExpCov : numpy.ndarray
    The expected covariance for each security.
state : numpy.ndarray
    The current state of the environment.
"""

import numpy as np
import gym
from gym import spaces

np.random.seed(123)


class TrainingEnv(gym.Env):

    def __init__(self):

        self.count_buyable_securities = 2

        obs_low = np.concatenate((np.array([0, 0, 0]),  # clientAge, portValue, target_spend_dollars
                                 np.zeros(self.count_buyable_securities)))  # security weights

        obs_high = np.concatenate((np.ones(3),  # clientAge, portValue, target_spend_dollars
                                  np.ones(self.count_buyable_securities)))  # security weights

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.count_buyable_securities,), dtype=np.float32)

        self.stepsPerYear = 2

        self.min_start_age = 60
        self.max_start_age = 100
        self.age_step_size = 1 / (self.stepsPerYear * (self.max_start_age - self.min_start_age))
        self.start_age = 0

        self.secExpRet = np.array([0, 0.07]) / self.stepsPerYear
        self.secExpCov = np.array([0,
                                   0, 0.04]) / self.stepsPerYear

    def step(self, action):

        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        invest_pct = action / np.maximum(np.sum(action), 0.000000001)

        if np.sum(invest_pct) > 1.0000001:
            raise NameError("invest_pct = " + str(invest_pct))

        if np.sum(invest_pct) < 0.9999999:
            raise NameError("invest_pct = " + str(invest_pct))

        sop_client_age, sop_port_value, target_spend_dollars = self.state[np.arange(3)]

        post_trade_pre_wd_security_dollars = sop_port_value * invest_pct

        ############################################################################
        # Set up post-withdrawal, pre-investment
        ############################################################################

        post_trade_post_wd_security_dollars = post_trade_pre_wd_security_dollars
        post_trade_post_wd_security_dollars[0] = post_trade_post_wd_security_dollars[0] - target_spend_dollars

        post_wd_port_value = np.sum(post_trade_post_wd_security_dollars)

        if post_wd_port_value < (1 / 1_000_000):
            # this may not be right....come back to it later
            post_trade_post_wd_security_weights = np.ones(self.count_buyable_securities) / self.count_buyable_securities
        else:
            post_trade_post_wd_security_weights = post_trade_post_wd_security_dollars / post_wd_port_value

        ############################################################################
        # Determine if we ran out of money
        # Assign rewards
        ############################################################################

        if post_trade_post_wd_security_dollars[0] < 0:

            done = True

            # reward = nTimesteps client is broke, scaled to -1 for self.min_start_age and 0 for self.max_start_age
            count_broke_timesteps = (1 - sop_client_age) / self.age_step_size
            max_steps = 1 / self.age_step_size
            reward = -(count_broke_timesteps / max_steps)

            eop_port_value = sop_port_value - target_spend_dollars

            eop_client_age = sop_client_age + self.age_step_size

            self.state = np.concatenate((np.array([eop_client_age, eop_port_value, target_spend_dollars]),
                                         post_trade_post_wd_security_weights))

            return np.array(self.state, dtype=np.float32), reward, done, {}

        else:
            reward = float(0.0)
            done = False

        ############################################################################
        # See how investment performed
        ############################################################################
        exp_cov_matrix = np.array([[0, 0],
                                  [0, 0.04]]) / self.stepsPerYear

        # Calculate investment return
        security_return = np.random.multivariate_normal(self.secExpRet, exp_cov_matrix, 1)[0]
        eop_dollars_security = post_trade_post_wd_security_dollars * (1 + security_return)
        eop_port_value = np.sum(eop_dollars_security)

        if eop_port_value < (1 / 1_000_000):
            eop_weight_security = np.ones(self.count_buyable_securities) / self.count_buyable_securities
        else:
            eop_weight_security = eop_dollars_security / eop_port_value

        eop_client_age = sop_client_age + self.age_step_size
        if eop_client_age >= 1.0:
            done = True

        self.state = np.concatenate((np.array([eop_client_age, eop_port_value, target_spend_dollars]),
                                     eop_weight_security))

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        age_reset = self.start_age

        starting_port_value_reset = 1

        target_spend_dollars_reset = 0.04

        sec_pct_reset = np.ones(self.count_buyable_securities) / self.count_buyable_securities

        self.state = np.concatenate((np.array([age_reset, starting_port_value_reset, target_spend_dollars_reset]),
                                     sec_pct_reset))

        return np.array(self.state, dtype=np.float32)
    
