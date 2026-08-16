import numpy as np

#Task 1

def load_data():
    """
    load data function
    """

    #allocation of data container
    data = {}

    #fill
    data['GDP'] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    return data


def process_data(data):
    """
    process data function
    """

    #verify data
    assert 'GDP' in data, "Data must contain 'GDP' key"

    #take log
    for k in ['GDP']:
        v = data[k]
        data[f'log_{k}'] = np.log(v)

    return data

# Task 2

# Task 2.1-2.2

class IncomeModelClass:
    """ Life-cycle income model used in Task 2. """

    def __init__(self, seed=1986):
        """ set up parameters for the model and simulation """

        #simulation numbers
        self.N = 50_000
        self.seed = seed
        self.ages = np.arange(18, 66)

        #education
        self.p_education = np.array([0.40, 0.35, 0.25])
        self.S = np.array([1, 3, 5])
        self.h0 = np.array([1.00, 1.20, 1.55])
        self.Delta = np.array([0.010, 0.020, 0.030])

        #human capital
        self.delta = 0.06
        self.sigma_psi = 0.10

        #labor market
        self.lam = 0.60
        self.sigma = 0.05

        #income
        self.y_SU = 0.45
        self.rho = 0.60
        self.y_floor = 0.35
    
    def simulate(self):
        "simulation of the model"

        #random number generator
        rng = np.random.default_rng(self.seed)

        #draw education
        education = rng.choice(
            3,
            size=self.N,
            p=self.p_education
        )

        self.education = education

        #find labor-market entry
        self.years_education = self.S[self.education] 
        self.age_entry = 18 + self.years_education

        #setup employment
        T = len(self.ages)
        self.employed = np.empty((self.N,T))

        #simulation of employment
        for i in range(self.N):

            for t,age in enumerate(self.ages):

                #in education or just entered labor market
                if age <= self.age_entry[i]:
                    self.employed[i,t] = 0

                #already on labor market
                else:

                    draw = rng.random()

                    #unemployed last year
                    if self.employed[i,t-1] == 0:

                        if draw < self.lam:
                            self.employed[i,t] = 1
                        else:
                            self.employed[i,t] = 0

                    #employed last year
                    else:

                        if draw < self.sigma:
                            self.employed[i,t] = 0
                        else:
                            self.employed[i,t] = 1

        #setup of human capital
        self.human_capital = np.empty((self.N,T))

        #simulation of capital
        for i in range(self.N):

            for t,age in enumerate(self.ages):

                #education and labor market entry
                if age <= self.age_entry[i]:

                    self.human_capital[i,t] = (
                        self.h0[self.education[i]]
                    )

                #after labor market entry
                else:

                    psi = rng.lognormal(
                        -0.5*self.sigma_psi**2,
                        self.sigma_psi
                    )

                    #employed last year
                    if self.employed[i,t-1] == 1:

                        self.human_capital[i,t] = (
                            self.human_capital[i,t-1]
                            * (1 + self.Delta[self.education[i]])
                            * psi
                        )

                    #unemployed last year
                    else:

                        self.human_capital[i,t] = (
                            self.human_capital[i,t-1]
                            * (1 - self.delta)
                            * psi
                        )

        #income setup
        self.income = np.empty((self.N,T))

        #simulation of income
        for i in range(self.N):

                    ever_employed = 0
                    last_job_income = 0

                    for t,age in enumerate(self.ages):

                        #in education
                        if age < self.age_entry[i]:

                            self.income[i,t] = self.y_SU

                        #employed
                        elif self.employed[i,t] == 1:

                            self.income[i,t] = self.human_capital[i,t]

                            last_job_income = self.income[i,t]
                            ever_employed = 1

                        #unemployed
                        else:

                            if ever_employed == 1:
                                self.income[i,t] = self.rho*last_job_income

                            else:
                                self.income[i,t] = self.y_floor

# Task 2.3

def lorenz_curve(x):
    """ calculate the Lorenz curve

    Args:

        x (ndarray): vector of incomes

    Returns:

        pop_share (ndarray): cumulative share of the population, from 0 to 1
        income_share (ndarray): cumulative share of total income, from 0 to 1

    """

    #sort incomes
    x = np.sort(x)
    N = len(x)

    #cumulative share of population
    pop_share = np.concatenate(([0.0], np.arange(1, N+1)/N))

    #cumulative share of income
    income_share = np.concatenate(([0.0], np.cumsum(x)/np.sum(x)))

    return pop_share, income_share

def gini(x):
    """ calculate the Gini coefficient

    Args:

        x (ndarray): vector of incomes

    Returns:

        gini (float): Gini coefficient

    """

    #sort incomes
    x = np.sort(x)

    #calculation of coefficient
    N = len(x)
    weighted_sum = 0
    total_income = 0

    for i in range(N):
        weighted_sum += (i+1)*x[i]
        total_income += x[i]

    gini = (
        2*weighted_sum/(N*total_income)
        - (N+1)/N
    )

    return gini
