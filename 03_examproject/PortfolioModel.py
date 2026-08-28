""" a portfolio with a risky and a safe asset (Problem 3)

Starting point for the exam. The methods raising NotImplementedError are the ones
you should write yourself.

"""

from types import SimpleNamespace
import numpy as np

class PortfolioModelClass:
    """ a portfolio of a risky and a safe asset with a rebalancing rule """

    def __init__(self,**kwargs):
        """ set the default parameters, then overwrite with any keyword arguments """

        par = self.par = SimpleNamespace()

        # a. returns
        par.mu = 0.05 # mean log return on the risky asset
        par.sigma = 0.20 # standard deviation of the log return on the risky asset
        par.r = 0.01 # log return on the safe asset

        # b. the rebalancing rule
        par.theta_star = 0.50 # target share of wealth in the risky asset
        par.Delta = 0.10 # width of the no-trade band
        par.tau = 0.01 # proportional transaction cost

        # c. preferences
        par.gamma = 3.0 # relative risk aversion

        # d. simulation settings
        par.W0 = 1.0 # initial wealth
        par.T = 40 # number of periods
        par.N = 50_000 # number of simulated portfolios
        par.seed = 2026 # seed for the random number generator

        # e. overwrite with keyword arguments, e.g. PortfolioModelClass(Delta=0.0)
        for key,value in kwargs.items(): setattr(par,key,value)

        # f. empty container for simulation results
        self.sim = SimpleNamespace()

    def __str__(self):
        """ called when using print """

        par = self.par

        text = 'Portfolio model with:\n'
        text += f'  mu    = {par.mu:.4f}, sigma = {par.sigma:.4f}, r = {par.r:.4f}\n'
        text += f'  theta_star = {par.theta_star:.4f}, Delta = {par.Delta:.4f}, tau = {par.tau:.4f}\n'
        text += f'  gamma = {par.gamma:.4f} (relative risk aversion)\n'
        text += f'  W0 = {par.W0:.2f}, T = {par.T}, N = {par.N:,}, seed = {par.seed}'

        return text

    def draw_returns(self):
        """ draw the gross return on the risky asset in all periods and all portfolios

        Returns:

            (ndarray): gross returns with shape (N,T)

        """

        par = self.par

        rng = np.random.default_rng(par.seed)
        eps = rng.normal(size=(par.N,par.T))

        return np.exp(par.mu + par.sigma*eps)

    def u(self,W):
        """ CRRA utility of wealth """

        par = self.par

        return W**(1-par.gamma)/(1-par.gamma)

    def trade(self,theta):
        """ the share of wealth in the risky asset after trading, and the amount traded

        Args:

            theta (ndarray): share of wealth in the risky asset before trading

        Returns:

            theta_post (ndarray): share of wealth in the risky asset after trading
            traded (ndarray): amount traded as a share of wealth

        """

        par = self.par

        # a. the no-trade band is left
        outside = np.abs(theta-par.theta_star) > par.Delta

        # b. trade all the way back to the target, or do nothing
        theta_post = np.where(outside,par.theta_star,theta)

        # c. the amount traded
        traded = np.abs(theta_post-theta)

        return theta_post,traded

    def simulate(self,R=None):
        """ simulate all N portfolios forward T periods

        Args:

            R (ndarray,optional): gross returns with shape (N,T), drawn if None

        Returns:

            (SimpleNamespace): the simulated paths, also stored in self.sim

        """

        par = self.par
        sim = self.sim

        if R is None: R = self.draw_returns()
        assert R.shape == (par.N,par.T), f'the returns must have shape {(par.N,par.T)}, but have {R.shape}'

        # a. the safe gross return
        Rf = np.exp(par.r)

        # b. allocate memory
        W = np.empty((par.N,par.T+1)) # wealth before trading
        theta = np.empty((par.N,par.T+1)) # share in the risky asset before trading
        traded = np.empty((par.N,par.T)) # amount traded

        # c. the investor starts at the target
        W[:,0] = par.W0
        theta[:,0] = par.theta_star

        # d. loop over time, vectorized over portfolios
        for t in range(par.T):

            # i. trade back to the target if the band is left
            theta_post,traded[:,t] = self.trade(theta[:,t])

            # ii. pay the transaction cost
            W_post = W[:,t]*(1-par.tau*traded[:,t])

            # iii. realize the returns
            W[:,t+1] = theta_post*W_post*R[:,t] + (1-theta_post)*W_post*Rf

            # iv. the share in the risky asset at the start of the next period
            theta[:,t+1] = theta_post*W_post*R[:,t]/W[:,t+1]

        # e. store the results
        sim.R = R
        sim.W = W
        sim.theta = theta
        sim.traded = traded

        return sim

    def summary(self):
        """ the numbers to report for a rule, including expected utility

        Returns:

            (SimpleNamespace): the six numbers, also stored in self.sim

        """

        par = self.par
        sim = self.sim

        # a. terminal wealth
        W_T = sim.W[:,-1]

        # b. the six numbers
        res = sim.res = SimpleNamespace()
        res.trades = np.mean(np.sum(sim.traded > 0,axis=1))
        res.distance = np.mean(np.abs(sim.theta[:,:par.T]-par.theta_star))
        res.mean = np.mean(W_T)
        res.median = np.median(W_T)
        res.p10 = np.percentile(W_T,10)
        res.utility = np.mean(self.u(W_T))

        return res