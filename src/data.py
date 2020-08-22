import pandas as pd
import numpy as np
import requests as rq


def expand(df):
    '''Fill missing dates in an irregular timeline'''
    min_date = df['date'].min()
    max_date = df['date'].max()
    idx = pd.date_range(min_date, max_date)
    
    df.index = pd.DatetimeIndex(df.date)
    df = df.drop(columns=['date'])
    return df.reindex(idx, method='pad').reset_index().rename(columns={'index':'date'})

def prefill(df, min_date):
    '''Fill zeros from first_case_date to df.date.min()'''
    assert(len(df.name.unique()) == 1)
    s = df.name.unique().item()
    min_date = min_date
    max_date = df['date'].max()
    idx = pd.date_range(min_date, max_date)
    
    df.index = pd.DatetimeIndex(df.date)
    df = df.drop(columns=['date'])
    return df.reindex(idx).reset_index().rename(columns={
        'index': 'date'
    }).fillna({
        'name': s,
        'confirmed': 0,
        'deceased': 0,
        'recovered': 0,
        'tested': 0
    })

def get_statewise_data():
    """get historic statewise covid data from covid19india API"""
    r=rq.get('https://api.covid19india.org/v3/min/timeseries.min.json')
    ts = r.json()

    data = []
    for state in ts:
        if state=='UN': continue
        for date in ts[state]:
            ttl = ts[state][date]['total']
            data.append((state, date, ttl.get('confirmed', 0), ttl.get('deceased', 0), ttl.get('recovered', 0), ttl.get('tested', 0)))

    states_df = pd.DataFrame(data, columns=['name', 'date', 'confirmed', 'deceased', 'recovered', 'tested'])
    states_df['state'] = states_df['name']
    states_df['date'] = pd.to_datetime(states_df['date'])
    return states_df


def get_districtwise_data():
    """get historic districtwise covid data from covid19india.org"""
    r=rq.get('https://api.covid19india.org/v4/data-all.json')
    respObj = r.json()

    ALL_STATES = list(respObj['2020-07-01'].keys())
    ALL_STATES.remove('UN')
    ALL_DISTTS = {}
    for st in ALL_STATES:
        if st == 'TT': continue
        ALL_DISTTS[st] = respObj['2020-07-01'][st]['districts'].keys()
        
    states_data = []
    distts_data = []
    for dt in respObj:
        for st in ALL_STATES:
            stateObj = respObj[dt].get(st, {})
            c = stateObj.get('total', {}).get('confirmed', 0)
            d = stateObj.get('total', {}).get('deceased', 0)
            r = stateObj.get('total', {}).get('recovered', 0)
            t = stateObj.get('total', {}).get('tested', 0)
            states_data.append((dt, st, c, d, r, t))
            if st == 'TT': continue
            for ds in ALL_DISTTS[st]:
                districtObj = stateObj.get('districts', {}).get(ds, {})
                c = districtObj.get('total', {}).get('confirmed', 0)
                d = districtObj.get('total', {}).get('deceased', 0)
                r = districtObj.get('total', {}).get('recovered', 0)
                t = districtObj.get('total', {}).get('tested', 0)
                distts_data.append((dt, st, ds, c, d, r, t))
                
    states_df = pd.DataFrame(states_data, columns=['date', 'state', 'confirmed', 'deceased', 'recovered', 'tested'])
    distts_df = pd.DataFrame(distts_data, columns=['date', 'state', 'name', 'confirmed', 'deceased', 'recovered', 'tested'])
    states_df['date'] = pd.to_datetime(states_df['date'])
    distts_df['date'] = pd.to_datetime(distts_df['date'])
    
    return distts_df


def fix_anomalies_owid(df):
    """fix anomalies in owid data frame"""
    # MH data fix: spread 17 Juns deaths over last 60 days
    if 'new_deaths' in df.columns:
        df.loc[(df['date']=='2020-06-17') & (df['location']=='India'), 'new_deaths'] = 353 #actual=2003
        t = np.random.rand(60)
        t = (2003-353) * t / sum(t)
        df.loc[(df['date']>='2020-04-18') & (df['date']<'2020-06-17') & (df['location']=='India'), 'new_deaths'] += np.int32(t)
    
    # remove countries with population_density and gdp = NaN
    countries_pd_nan = df.loc[df['population_density'].isna()].location.unique()
    df = df.loc[~df.location.isin(countries_pd_nan)]
    countries_gdp_nan = df.loc[df['gdp_per_capita'].isna()].location.unique()
    df = df.loc[~df.location.isin(countries_gdp_nan)]

    return df
