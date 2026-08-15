import numpy as np

#Task 1

def load_data():
    """
    load data function
    """

    # a. allocate data container
    data = {}

    # b. fill
    data['GDP'] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    return data


def process_data(data):
    """
    process data function
    """

    # a. verify data
    assert 'GDP' in data, "Data must contain 'GDP' key"

    # b. take log
    for k in ['GDP']:
        v = data[k]
        data[f'log_{k}'] = np.log(v)

    return data

#task 2 

#Task 2.1-2.2

class IncomeModelClass:
    """Life-cycle model of income."""

    def __init__(self, seed=1986):
        """Setup model parameters."""

        #simulation numbers
        self.N = 50_000
        self.seed = seed
        self.ages = np.arange(18, 66)

        #education numbers
        self.p_education = np.array([0.40, 0.35, 0.25])
        self.S = np.array([1, 3, 5])
        self.h0 = np.array([1.00, 1.20, 1.55])
        self.Delta = np.array([0.010, 0.020, 0.030])

        #human capital numbers
        self.delta = 0.06
        self.sigma_psi = 0.10

        #labor market numbers 
        self.lam = 0.60
        self.sigma = 0.05

        #income numbers
        self.y_SU = 0.45
        self.rho = 0.60
        self.y_floor = 0.35
    
    def simulate(self):

        rng = np.random.default_rng(self.seed)

        #find education
        education = rng.choice(
            3,
            size=self.N,
            p=self.p_education
        )

        self.education = education

        #years of education
        self.years_education = self.S[self.education] 

        #age when entering labor market
        self.age_entry = 18 + self.years_education

        # d. allocate employment array
        # d. setup employment
        T = len(self.ages)
        self.employed = np.empty((self.N,T))

        # e. simulate employment
        for i in range(self.N):

            for t,age in enumerate(self.ages):

                # i. in education or just entered labor market
                if age <= self.age_entry[i]:
                    self.employed[i,t] = 0

                # ii. already on labor market
                else:

                    draw = rng.random()

                    # unemployed last year
                    if self.employed[i,t-1] == 0:

                        if draw < self.lam:
                            self.employed[i,t] = 1
                        else:
                            self.employed[i,t] = 0

                    # employed last year
                    else:

                        if draw < self.sigma:
                            self.employed[i,t] = 0
                        else:
                            self.employed[i,t] = 1

        # f. setup human capital
        self.human_capital = np.empty((self.N,T))

        # g. simulate human capital
        for i in range(self.N):

            for t,age in enumerate(self.ages):

                # i. education and labor market entry
                if age <= self.age_entry[i]:

                    self.human_capital[i,t] = (
                        self.h0[self.education[i]]
                    )

                # ii. after labor market entry
                else:

                    psi = rng.lognormal(
                        -0.5*self.sigma_psi**2,
                        self.sigma_psi
                    )

                    # employed last year
                    if self.employed[i,t-1] == 1:

                        self.human_capital[i,t] = (
                            self.human_capital[i,t-1]
                            * (1 + self.Delta[self.education[i]])
                            * psi
                        )

                    # unemployed last year
                    else:

                        self.human_capital[i,t] = (
                            self.human_capital[i,t-1]
                            * (1 - self.delta)
                            * psi
                        )

        # h. setup income
        self.income = np.empty((self.N,T))

        # i. simulate income
        for i in range(self.N):

                    ever_employed = 0
                    last_job_income = 0

                    for t,age in enumerate(self.ages):

                        # i. in education
                        if age < self.age_entry[i]:

                            self.income[i,t] = self.y_SU

                        # ii. employed
                        elif self.employed[i,t] == 1:

                            self.income[i,t] = self.human_capital[i,t]

                            last_job_income = self.income[i,t]
                            ever_employed = 1

                        # iii. unemployed
                        else:

                            if ever_employed == 1:
                                self.income[i,t] = self.rho*last_job_income

                            else:
                                self.income[i,t] = self.y_floor