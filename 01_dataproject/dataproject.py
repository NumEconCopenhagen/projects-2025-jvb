import numpy as np
from dstapi import DstApi
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