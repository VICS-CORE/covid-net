import pandas as pd
import numpy as np
import requests as rq


def get_statewise_data():
    """get historic statewise covid data from covid19india API"""
    r=rq.get('https://api.covid19india.org/v3/min/timeseries.min.json')
    ts = r.json()

    data = []
    for state in ts:
        for date in ts[state]:
            ttl = ts[state][date]['total']
            data.append((state, date, ttl.get('confirmed', 0), ttl.get('deceased', 0), ttl.get('recovered', 0), ttl.get('tested', 0)))

    states_df = pd.DataFrame(data, columns=['state', 'date', 'confirmed', 'deceased', 'recovered', 'tested'])
    states_df['date'] = pd.to_datetime(states_df['date'])
    return states_df

def fix_anomalies_owid(df):
    """fix anomalies in owid data frame"""
    # MH data fix: spread 17 Juns deaths over last 60 days
    if 'new_deaths' in df.columns:
        df.loc[(df['date']=='2020-06-17') & (df['location']=='India'), 'new_deaths'] = 353 #actual=2003
        t = np.random.rand(60)
        t = (2003-353) * t / sum(t)
        df.loc[(df['date']>='2020-04-18') & (df['date']<'2020-06-17') & (df['location']=='India'), 'new_deaths'] += np.int32(t)
    
    # remove countries with population_density NaN
    countries_pd_nan = df.loc[df['population_density'].isna()].location.unique()
    df = df.loc[~df.location.isin(countries_pd_nan)]

    return df
