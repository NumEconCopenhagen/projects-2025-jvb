import numpy as np
from dstapi import DstApi
import pandas as pd


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
def kappa(k, alpha):
    """
    Weight function for instantaneous inflation, kappa(k, alpha).
    k can be a single number or an array of integers from 0 to 11.
    """

    # a. ensure that k is an array so the function works vectorized
    k = np.asarray(k, dtype=float)

    # b. the denominator always sums over all 12 possible k-values (0 to 11)
    all_k = np.arange(12)
    denominator = np.sum((12 - all_k)**alpha)

    # c. the weight itself, scaled by 12 (see the formula in the assignment)
    return (12 - k)**alpha / denominator * 12


def instantaneous_inflation(window, alpha):
    """
    Computes pi^{12, alpha}_t for one rolling window of 12 monthly
    inflation rates. Used together with .rolling(12).apply(...).

    window[0] is the oldest month in the window (t-11), while window[-1]
    is the newest (t) -- so k=11 corresponds to window[0] and k=0 to window[-1].
    """

    # a. construct k in the same order as the window: 11, 10, ..., 1, 0
    k = np.arange(11, -1, -1)

    # b. obtain the corresponding weights
    w = kappa(k, alpha)

    # c. det vejede produkt
    return np.prod((1 + window)**w) - 1


def add_instantaneous_inflation(data, alphas=[0,1,2,3]):
    """
    Adds a pi_12_alpha{a} column to the data for each alpha in alphas.
    """
    for a in alphas:
        data[f'pi_12_alpha{a}'] = data['inflation_mom'].rolling(12).apply(
            lambda w: instantaneous_inflation(w, a), raw=True
        )
    return data

def load_core_inflation_data():
    """
    Loads 12-month inflation for total CPI, CPI excluding energy, and
    core inflation (excluding energy and unprocessed food) from PRIS111.
    """

    # a. connect to the table
    pris111 = DstApi('PRIS111')

    # b. build the parameters
    params = pris111.define_base_params(language='en')
    params['variables'][0]['values'] = ['000000', '151000', '141000']  # VAREGR
    params['variables'][1]['values'] = ['300']                          # ENHED: 12-month pct. change
    params['variables'][2]['values'] = ['*']                            # Time: all months

    # c. load the data
    data_core = pris111.get_data(params=params)

    return data_core


def process_core_inflation_data(data_core):
    """
    Creates a date variable and converts INDHOLD to float.
    """

    # a. date variable
    data_core['date'] = pd.to_datetime(data_core['TID'], format='%YM%m')

    # b. ensure numeric type (the same issue as with PRIS113 earlier)
    data_core['INDHOLD'] = pd.to_numeric(data_core['INDHOLD'], errors='coerce')

    # c. sort chronologically within each commodity group
    data_core = data_core.sort_values(['VAREGR','date']).reset_index(drop=True)

    return data_core

import re

def get_4digit_categories(pris111):
    """
    Finds all VAREGR categories at the four-digit level (e.g. '01.1.1.1 Rice').
    """

    # a. load all VAREGR levels
    levels = pris111.variable_levels('VAREGR', language='en')

    # b. match labels of the form XX.X.X.X (four numbers separated by dots)
    mask = levels['text'].str.match(r'^\d{2}\.\d+\.\d+\.\d+\s')

    return levels[mask]


def load_disaggregated_inflation(pris111, categories_4digit):
    """
    Loads 12-month inflation for all four-digit product categories.
    """

    # a. build the parameters -- use the list of IDs as values
    params = pris111.define_base_params(language='en')
    params['variables'][0]['values'] = categories_4digit['id'].tolist()  # VAREGR
    params['variables'][1]['values'] = ['300']                            # ENHED: 12-month pct. change
    params['variables'][2]['values'] = ['*']                              # Tid

    # b. load the data
    data_disagg = pris111.get_data(params=params)

    # c. date variable and numeric type (as above)
    data_disagg['date'] = pd.to_datetime(data_disagg['TID'], format='%YM%m')
    data_disagg['INDHOLD'] = pd.to_numeric(data_disagg['INDHOLD'], errors='coerce')

    return data_disagg


def compute_percentiles(data_disagg):
    """
    Computes the 25th, 50th, and 75th percentiles of 12-month inflation
    across categories for each month.
    """
    percentiles = data_disagg.groupby('date')['INDHOLD'].agg(
        p25=lambda x: x.quantile(0.25),
        p50=lambda x: x.quantile(0.50),
        p75=lambda x: x.quantile(0.75)
    ).reset_index()

    return percentiles

def load_disaggregated_index(pris111, categories_4digit):
    """
    Loads the price index itself (the level, not inflation) for the four-digit
    categories -- used to compute the change from August 2020 to August 2025.
    """

    params = pris111.define_base_params(language='en')
    params['variables'][0]['values'] = categories_4digit['id'].tolist()  # VAREGR
    params['variables'][1]['values'] = ['100']                            # ENHED: Index
    params['variables'][2]['values'] = ['*']                              # Tid

    data_idx = pris111.get_data(params=params)

    data_idx['date'] = pd.to_datetime(data_idx['TID'], format='%YM%m')
    data_idx['INDHOLD'] = pd.to_numeric(data_idx['INDHOLD'], errors='coerce')

    return data_idx


def compute_pct_change_period(data_idx, start='2020-08-01', end='2025-08-01'):
    """
    Computes the percentage change in the price index for each category
    mellem start og end.
    """

    # a. values at the start and end dates, by category
    start_vals = data_idx[data_idx['date'] == start].set_index('VAREGR')['INDHOLD']
    end_vals = data_idx[data_idx['date'] == end].set_index('VAREGR')['INDHOLD']

    # b. percentage change
    pct_change = (end_vals / start_vals - 1) * 100

    return pct_change.dropna()

def top_bottom_categories(pct_change_period, n=10):
    """
    Finds the n categories with the largest and smallest price increases.
    """
    top = pct_change_period.sort_values(ascending=False).head(n)
    bottom = pct_change_period.sort_values(ascending=True).head(n)

    return top, bottom
