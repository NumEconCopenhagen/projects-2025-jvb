import pandas as pd
from states import STATES


def download_state_data(fred, states=STATES):
    "Download real GDP and population for all of the states in states.py"

    gdp = {}
    population = {}

    for state in states:
        gdp[state] = fred.get_series(f'{state}RGSP')
        population[state] = fred.get_series(f'{state}POP')

    gdp = pd.DataFrame(gdp)
    population = pd.DataFrame(population)

    gdp.index = gdp.index.year
    population.index = population.index.year
    gdp.index.name = 'year'
    population.index.name = 'year'

    return gdp, population


def clean_state_data(gdp, population, states=STATES):
    "Keep common years and states with complete GDP and population data."

    common_years = gdp.index.intersection(population.index).sort_values()
    gdp = gdp.loc[common_years, states].copy()
    population = population.loc[common_years, states].copy()

    complete_states = [
        state
        for state in states
        if not gdp[state].isna().any() and not population[state].isna().any()
    ]

    gdp = gdp.loc[:, complete_states]
    population = population.loc[:, complete_states]

    return gdp, population


def endpoint_summary(y, years):
    "Summarize the highest, lowest, average, and ratio for selected years."

    rows = []
    for year in years:
        values = y.loc[year]
        highest_state = values.idxmax()
        lowest_state = values.idxmin()

        rows.append({
            'year': year,
            'highest state': highest_state,
            'highest': values.loc[highest_state],
            'lowest state': lowest_state,
            'lowest': values.loc[lowest_state],
            'highest / lowest': values.max() / values.min(),
            'average': values.mean(),
        })

    return pd.DataFrame(rows).set_index('year')
