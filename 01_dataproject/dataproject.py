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
    Vægtfunktion til instantaneous inflation, kappa(k,alpha).
    k kan være et enkelt tal eller et array af heltal fra 0 til 11.
    """

    # a. sørg for at k er et array, så funktionen virker vektoriseret
    k = np.asarray(k, dtype=float)

    # b. nævneren summeres altid over alle 12 mulige k-værdier (0 til 11)
    all_k = np.arange(12)
    denominator = np.sum((12 - all_k)**alpha)

    # c. selve vægten, skaleret med 12 (se opgavetekstens formel)
    return (12 - k)**alpha / denominator * 12


def instantaneous_inflation(window, alpha):
    """
    Beregner pi^{12,alpha}_t for ét rullende vindue af 12 månedlige
    inflationsrater. Bruges sammen med .rolling(12).apply(...).

    window[0] er den ældste måned i vinduet (t-11), window[-1] er den
    nyeste (t) -- dvs. k=11 svarer til window[0], k=0 svarer til window[-1].
    """

    # a. byg k i samme rækkefølge som vinduet: 11,10,...,1,0
    k = np.arange(11, -1, -1)

    # b. hent tilhørende vægte
    w = kappa(k, alpha)

    # c. det vejede produkt
    return np.prod((1 + window)**w) - 1


def add_instantaneous_inflation(data, alphas=[0,1,2,3]):
    """
    Tilføjer en kolonne pi_12_alpha{a} til data for hver alpha i alphas.
    """
    for a in alphas:
        data[f'pi_12_alpha{a}'] = data['inflation_mom'].rolling(12).apply(
            lambda w: instantaneous_inflation(w, a), raw=True
        )
    return data

def load_core_inflation_data():
    """
    Henter 12-måneders inflation for total-CPI, CPI ekskl. energi, og
    kerneinflation (ekskl. energi og uforarbejdede fødevarer) fra PRIS111.
    """

    # a. opret forbindelse til tabellen
    pris111 = DstApi('PRIS111')

    # b. byg parametrene
    params = pris111.define_base_params(language='en')
    params['variables'][0]['values'] = ['000000', '151000', '141000']  # VAREGR
    params['variables'][1]['values'] = ['300']                          # ENHED: 12-month pct. change
    params['variables'][2]['values'] = ['*']                            # Tid: alle måneder

    # c. hent data
    data_core = pris111.get_data(params=params)

    return data_core


def process_core_inflation_data(data_core):
    """
    Laver dato-variabel og konverterer INDHOLD til float.
    """

    # a. dato-variabel
    data_core['date'] = pd.to_datetime(data_core['TID'], format='%YM%m')

    # b. sørg for numerisk type (samme fejl som med PRIS113 tidligere)
    data_core['INDHOLD'] = pd.to_numeric(data_core['INDHOLD'], errors='coerce')

    # c. sortér kronologisk inden for hver varegruppe
    data_core = data_core.sort_values(['VAREGR','date']).reset_index(drop=True)

    return data_core

import re

def get_4digit_categories(pris111):
    """
    Finder alle VAREGR-kategorier på 4-cifret niveau (fx '01.1.1.1 Rice').
    """

    # a. hent alle niveauer af VAREGR
    levels = pris111.variable_levels('VAREGR', language='en')

    # b. match tekster på formen XX.X.X.X (fire tal adskilt af punktum)
    mask = levels['text'].str.match(r'^\d{2}\.\d+\.\d+\.\d+\s')

    return levels[mask]


def load_disaggregated_inflation(pris111, categories_4digit):
    """
    Henter 12-måneders inflation for alle 4-cifrede produktkategorier.
    """

    # a. byg parametrene -- brug listen af id'er som values
    params = pris111.define_base_params(language='en')
    params['variables'][0]['values'] = categories_4digit['id'].tolist()  # VAREGR
    params['variables'][1]['values'] = ['300']                            # ENHED: 12-month pct. change
    params['variables'][2]['values'] = ['*']                              # Tid

    # b. hent data
    data_disagg = pris111.get_data(params=params)

    # c. dato-variabel og numerisk type (samme som tidligere)
    data_disagg['date'] = pd.to_datetime(data_disagg['TID'], format='%YM%m')
    data_disagg['INDHOLD'] = pd.to_numeric(data_disagg['INDHOLD'], errors='coerce')

    return data_disagg


def compute_percentiles(data_disagg):
    """
    Beregner 25., 50. og 75. percentil af 12-måneders inflation på tværs
    af kategorier, for hver måned.
    """
    percentiles = data_disagg.groupby('date')['INDHOLD'].agg(
        p25=lambda x: x.quantile(0.25),
        p50=lambda x: x.quantile(0.50),
        p75=lambda x: x.quantile(0.75)
    ).reset_index()

    return percentiles

def load_disaggregated_index(pris111, categories_4digit):
    """
    Henter selve prisindekset (niveau, ikke inflation) for de 4-cifrede
    kategorier -- bruges til at beregne ændringen fra aug 2020 til aug 2025.
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
    Beregner den procentvise ændring i prisindeks for hver kategori
    mellem start og end.
    """

    # a. værdier ved start- og slutdato, pr. kategori
    start_vals = data_idx[data_idx['date'] == start].set_index('VAREGR')['INDHOLD']
    end_vals = data_idx[data_idx['date'] == end].set_index('VAREGR')['INDHOLD']

    # b. procentvis ændring
    pct_change = (end_vals / start_vals - 1) * 100

    return pct_change.dropna()

def top_bottom_categories(pct_change_period, n=10):
    """
    Finder de n kategorier med størst og mindst prisstigning.
    """
    top = pct_change_period.sort_values(ascending=False).head(n)
    bottom = pct_change_period.sort_values(ascending=True).head(n)

    return top, bottom