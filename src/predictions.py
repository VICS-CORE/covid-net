import datetime as dt
import pandas as pd
import numpy as np
import json
import torch

from matplotlib.dates import DayLocator, AutoDateLocator, ConciseDateFormatter

DEVICE='cpu'

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
    assert(len(df.state.unique()) == 1)
    s = df.state.unique().item()
    min_date = min_date
    max_date = df['date'].max()
    idx = pd.date_range(min_date, max_date)
    
    df.index = pd.DatetimeIndex(df.date)
    df = df.drop(columns=['date'])
    return df.reindex(idx).reset_index().rename(columns={
        'index': 'date'
    }).fillna({
        'state': s,
        'confirmed': 0,
        'deceased': 0,
        'recovered': 0,
        'tested': 0
    })

def generate(states_df, STT_INFO, model, cp, feature, n_days_prediction, prediction_offset, plot=False):
    """
    Generates predictions, given states data, a model and checkpoint dictionary
    states_df: dataframe with current data for all states.
    STT_INFO: static dictionary with info for all states
    model: CovidNet model object
    cp: checkpoint dictionary returned from load_checkpoint
    feature: 0 for confirmed, 1 for deaths etc.
    n_days_prediction: number of days to predict
    prediction_offset: number of days from real data to be skipped
    plot(False): whether to plot charts and print logs
    """
    IP_SEQ_LEN = cp['config']['DS']['IP_SEQ_LEN']
    OP_SEQ_LEN = cp['config']['DS']['OP_SEQ_LEN']

    first_case_date = states_df['date'].min()
    n_days_data = len(expand(states_df.loc[states_df['state']=='TT']))
    assert(n_days_prediction%OP_SEQ_LEN == 0)

    agg_days = n_days_data - prediction_offset + n_days_prediction # number of days for plotting agg curve i.e. prediction + actual data 
    states_agg = np.zeros(agg_days)
    
    ax = None
    api = {}
    for state in STT_INFO:
        pop_fct = STT_INFO[state]["popn"] / 1000

        state_df = states_df.loc[states_df['state']==state][:-prediction_offset] # skip todays data. covid19 returns incomplete.
        state_df = prefill(expand(state_df), first_case_date)
        state_df['new_cases'] = state_df['confirmed'] - state_df['confirmed'].shift(1).fillna(0)
        state_df['new_deaths'] = state_df['deceased'] - state_df['deceased'].shift(1).fillna(0)
        state_df['new_recovered'] = state_df['recovered'] - state_df['recovered'].shift(1).fillna(0)
        state_df['new_tests'] = state_df['tested'] - state_df['tested'].shift(1).fillna(0)
        test_data = np.array(state_df[cp['config']['DS']['FEATURES']].rolling(7, center=True, min_periods=1).mean() / pop_fct, dtype=np.float32)

        in_data = test_data[-IP_SEQ_LEN:, cp['config']['IP_FEATURES']]
        out_data = np.ndarray(shape=(0, len(cp['config']['OP_FEATURES'])), dtype=np.float32)
        for i in range(int(n_days_prediction / OP_SEQ_LEN)):
            ip = torch.tensor(
                in_data,
                dtype=torch.float32
            ).to(DEVICE)
            try:
                pred = model.predict(ip.view(-1, IP_SEQ_LEN, len(cp['config']['IP_FEATURES']))).view(OP_SEQ_LEN, len(cp['config']['OP_FEATURES']))
            except Exception as e:
                print(state, e)
            in_data = np.append(in_data[-IP_SEQ_LEN+OP_SEQ_LEN:, :], pred.cpu().numpy(), axis=0)
            out_data = np.append(out_data, pred.cpu().numpy(), axis=0)

        sn = STT_INFO[state]['name']
        orig_df = pd.DataFrame({
            'actual': np.array(test_data[:,feature] * pop_fct, dtype=np.int)
        })
        fut_df = pd.DataFrame({
            'predicted': np.array(out_data[:,feature] * pop_fct, dtype=np.int)
        })
        # print(fut_df.to_csv(sep='|'))
        full_df = orig_df.append(fut_df, ignore_index=True, sort=False)
        full_df[sn] = full_df['actual'].fillna(0) + full_df['predicted'].fillna(0)
        full_df['total'] = full_df[sn].cumsum()

        states_agg += np.array(full_df[sn][-agg_days:].fillna(0))

        # generate date col for full_df from state_df
        start_date = state_df['date'].iloc[0]
        full_df['Date'] = pd.to_datetime([(start_date + dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(full_df))])
        # if full_df[sn].max() < 10000: # or full_df[sn].max() < 5000:
        #     continue

        # print state, cumulative, peak
        peak = full_df.loc[full_df[sn].idxmax()]
        if plot: print(sn, "|", peak['Date'].strftime("%b %d"), "|", int(peak[sn]), "|", int(full_df['total'].iloc[-1]))

        # export data for API
        full_df['daily_deaths'] = full_df[sn] * 0.028
        full_df['daily_recovered'] = full_df[sn].shift(14, fill_value=0) - full_df['daily_deaths'].shift(7, fill_value=0)
        full_df['daily_active'] = full_df[sn] - full_df['daily_recovered'] - full_df['daily_deaths']

        api[state] = {}
        for idx, row in full_df[-agg_days:].iterrows():
            row_date = row['Date'].strftime("%Y-%m-%d")
            api[state][row_date] = {
                "delta": {
                    "confirmed": int(row[sn]),
                    "deceased": int(row['daily_deaths']),
                    "recovered": int(row['daily_recovered']),
                    "active": int(row['daily_active'])
                }
            }
        
        # plot state chart
        if plot: 
            ax = full_df.plot(
                x='Date',
                y=[sn],
                title='Daily Cases',
                figsize=(15,10),
                grid=True,
                ax=ax,
                lw=3
            )
            mn_l = DayLocator()
            ax.xaxis.set_minor_locator(mn_l)
            mj_l = AutoDateLocator()
            mj_f = ConciseDateFormatter(mj_l, show_offset=False)
            ax.xaxis.set_major_formatter(mj_f)

    # plot aggregate chart
    if plot: 
        agg_df = pd.DataFrame({
            'states_agg': states_agg 
        })
        last_date = full_df['Date'].iloc[-1].to_pydatetime()
        start_date = last_date - dt.timedelta(days=agg_days)
        agg_df['Date'] = pd.to_datetime([(start_date + dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(agg_df))])
        ax = agg_df.plot(
            x='Date',
            y=['states_agg'],
            title='Aggregate daily cases',
            figsize=(15,10),
            grid=True,
            lw=3
        )
        mn_l = DayLocator()
        ax.xaxis.set_minor_locator(mn_l)
        mj_l = AutoDateLocator()
        mj_f = ConciseDateFormatter(mj_l, show_offset=False)
        ax.xaxis.set_major_formatter(mj_f)

        # plot peak in agg
        peakx = 178
        peak = agg_df.iloc[peakx]
        peak_desc = peak['Date'].strftime("%d-%b") + "\n" + str(int(peak['states_agg']))
        _ = ax.annotate(
            peak_desc, 
            xy=(peak['Date'] + dt.timedelta(days=1), peak['states_agg']),
            xytext=(peak['Date'] + dt.timedelta(days=45), peak['states_agg'] * .9),
            arrowprops={},
            bbox={'facecolor':'white'}
        )
        _ = ax.axvline(x=peak['Date'], linewidth=1, color='r')

    return api

def export_tracker(api, fn="predictions.json"):
    # aggregate predictions
    api['TT'] = {}
    for state in api:
        if state == 'TT':
            continue
        for date in api[state]:
            api['TT'][date] = api['TT'].get(date, {'delta':{}, 'total':{}})
            for k in ['delta']: #'total'
                api['TT'][date][k]['confirmed'] = api['TT'][date][k].get('confirmed', 0) + api[state][date][k]['confirmed']
                api['TT'][date][k]['deceased'] = api['TT'][date][k].get('deceased', 0) + api[state][date][k]['deceased']
                api['TT'][date][k]['recovered'] = api['TT'][date][k].get('recovered', 0) + api[state][date][k]['recovered']
                api['TT'][date][k]['active'] = api['TT'][date][k].get('active', 0) + api[state][date][k]['active']

    # export
    with open(fn, "w") as f:
        f.write(json.dumps(api, sort_keys=True))

def export_videoplayer(api, prediction_date, fn=""):
    # aggregate predictions
    api['TT'] = {}
    for state in api:
        if state == 'TT':
            continue
        for date in api[state]:
            api['TT'][date] = api['TT'].get(date, {})
            api['TT'][date]['c'] = api['TT'][date].get('c', 0) + api[state][date]['delta']['confirmed']
            api['TT'][date]['d'] = api['TT'][date].get('d', 0) + api[state][date]['delta']['deceased']
            api['TT'][date]['r'] = api['TT'][date].get('r', 0) + api[state][date]['delta']['recovered']
            api['TT'][date]['a'] = api['TT'][date].get('a', 0) + api[state][date]['delta']['active']

    # read previous and export
    try:
        with open(fn, "r") as f:
            out = json.loads(f.read())
    except Exception as e:
        out = {}

    with open(fn, "w") as f:
        out[prediction_date] = {'TT': api['TT']}
        f.write(json.dumps(out, sort_keys=True))
