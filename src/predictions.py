import datetime as dt
import pandas as pd
import numpy as np
import json
import torch

from matplotlib.dates import DayLocator, AutoDateLocator, ConciseDateFormatter

DEVICE='cpu'

def plot_linechart(df, xcol, ycol, ax=None, title="", show_peak=True):
    ax = df.plot(
        x=xcol,
        y=[ycol],
        title=title,
        figsize=(8,5),
        grid=True,
        ax=ax,
        lw=3
    )
    mn_l = DayLocator()
    ax.xaxis.set_minor_locator(mn_l)
    mj_l = AutoDateLocator()
    mj_f = ConciseDateFormatter(mj_l, show_offset=False)
    ax.xaxis.set_major_formatter(mj_f)
    ax.legend(loc=2)

    if show_peak:
        # plot peak in agg
        peakx = df[ycol].idxmax()
        peak = df.loc[peakx]
        peak_desc = peak[xcol].strftime("%d-%b") + "\n" + str(int(peak[ycol]))
        _ = ax.annotate(
            peak_desc, 
            xy=(peak[xcol] + dt.timedelta(days=1), peak[ycol]),
            xytext=(peak[xcol] + dt.timedelta(days=45), peak[ycol] * .9),
            arrowprops={},
            bbox={'facecolor':'white'}
        )
        _ = ax.axvline(x=peak[xcol], linewidth=1, color='r')
    
    return ax

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

def generate(df, INFO, model, cp, feature, n_days_prediction, prediction_offset, plot=False):
    """
    Generates predictions, given states data, a model and checkpoint dictionary
    df: dataframe with current data for all states/districts.
    INFO: static dictionary with info for all states/districts
    model: CovidNet model object
    cp: checkpoint dictionary returned from load_checkpoint
    feature: 0 for confirmed, 1 for deaths etc.
    n_days_prediction: number of days to predict
    prediction_offset: number of days from real data to be skipped
    plot(False): whether to plot charts and print logs
    """
    IP_SEQ_LEN = cp['config']['DS']['IP_SEQ_LEN']
    OP_SEQ_LEN = cp['config']['DS']['OP_SEQ_LEN']

    first_case_date = df['date'].min()
    n_days_data = len(df.loc[df.state=='KL'].date.unique())
    assert(n_days_prediction%OP_SEQ_LEN == 0)

    agg_days = n_days_data - prediction_offset + n_days_prediction # number of days for plotting agg curve i.e. prediction + actual data 

    api = {}
    states_agg = np.zeros(agg_days)
    state_ax = None
    for state in INFO:
        child_ax = None
        api[state] = {}
        child_agg = np.zeros(agg_days)
        for child in INFO[state]:
            pop_fct = child["popn"] / 1000
            ip_aux = torch.tensor(
                np.array(child["population_density"]),
                dtype=torch.float32
            ).to(DEVICE) if "population_density" in child else None

            child_df = df.loc[(df['state']==state) & (df['name']==child['name'])][:-prediction_offset] # skip todays data. covid19 returns incomplete.
            child_df = prefill(expand(child_df), first_case_date)
            child_df['new_cases'] = child_df['confirmed'] - child_df['confirmed'].shift(1).fillna(0)
            child_df['new_deaths'] = child_df['deceased'] - child_df['deceased'].shift(1).fillna(0)
            child_df['new_recovered'] = child_df['recovered'] - child_df['recovered'].shift(1).fillna(0)
            child_df['new_tests'] = child_df['tested'] - child_df['tested'].shift(1).fillna(0)
            test_data = np.array(child_df[cp['config']['DS']['FEATURES']].rolling(7, center=True, min_periods=1).mean() / pop_fct, dtype=np.float32)

            in_data = test_data[-IP_SEQ_LEN:, cp['config']['IP_FEATURES']]
            out_data = np.ndarray(shape=(0, len(cp['config']['OP_FEATURES'])), dtype=np.float32)
            for i in range(int(n_days_prediction / OP_SEQ_LEN)):
                ip = torch.tensor(
                    in_data,
                    dtype=torch.float32
                ).to(DEVICE)
                try:
                    args = [ip.view(-1, IP_SEQ_LEN, len(cp['config']['IP_FEATURES']))]
                    if len(cp['config'].get('AUX_FEATURES', [])):
                        args.append(ip_aux.view(1, len(cp['config']['AUX_FEATURES'])))
                    pred = model.predict(*args).view(OP_SEQ_LEN, len(cp['config']['OP_FEATURES']))
                except Exception as e:
                    print(state, e)
                if IP_SEQ_LEN == OP_SEQ_LEN:
                    in_data = pred.cpu().numpy()
                else:
                    in_data = np.append(in_data[-IP_SEQ_LEN+OP_SEQ_LEN:, :], pred.cpu().numpy(), axis=0)
                out_data = np.append(out_data, pred.cpu().numpy(), axis=0)

            cn = child['name']
            orig_df = pd.DataFrame({
                'actual': np.array(test_data[:,feature] * pop_fct, dtype=np.int)
            })
            fut_df = pd.DataFrame({
                'predicted': np.array(out_data[:,feature] * pop_fct, dtype=np.int)
            })
            # print(fut_df.to_csv(sep='|'))
            full_df = orig_df.append(fut_df, ignore_index=True, sort=False)
            full_df[cn] = full_df['actual'].fillna(0) + full_df['predicted'].fillna(0)
            full_df['total'] = full_df[cn].cumsum()

            child_agg += np.array(full_df[cn][-agg_days:].fillna(0))

            # generate date col for full_df from child_df
            start_date = child_df['date'].iloc[0]
            full_df['Date'] = pd.to_datetime([(start_date + dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(full_df))])
            # if full_df[cn].max() < 10000: # or full_df[cn].max() < 5000:
            #     continue

            # print state, cumulative, peak
            peak = full_df.loc[full_df[cn].idxmax()]
            if plot: print(state, "|", cn, "|", peak['Date'].strftime("%b %d"), "|", int(peak[cn]), "|", int(full_df['total'].iloc[-1]))

            # export data for API
            full_df['daily_deaths'] = full_df[cn] * 0.015
            full_df['daily_recovered'] = full_df[cn].shift(14, fill_value=0) - full_df['daily_deaths'].shift(7, fill_value=0)
            full_df['daily_active'] = full_df[cn] - full_df['daily_recovered'] - full_df['daily_deaths']

            api[state][cn] = {}
            for idx, row in full_df[-agg_days:].iterrows():
                row_date = row['Date'].strftime("%Y-%m-%d")
                api[state][cn][row_date] = {
                    "delta": {
                        "confirmed": int(row[cn]),
                        "deceased": int(row['daily_deaths']),
                        "recovered": int(row['daily_recovered']),
                        "active": int(row['daily_active'])
                    }
                }

            # plot individual chart
            if plot:
                child_ax = full_df.plot(
                    x='Date',
                    y=[cn],
                    title='Daily Cases for ' + state,
                    figsize=(8,5),
                    grid=True,
                    ax=child_ax,
                    lw=3
                )
                child_ax.legend(loc=2)
                mn_l = DayLocator()
                child_ax.xaxis.set_minor_locator(mn_l)
                mj_l = AutoDateLocator()
                mj_f = ConciseDateFormatter(mj_l, show_offset=False)
                child_ax.xaxis.set_major_formatter(mj_f)

        states_agg += child_agg
        # plot aggregate chart for children
        if plot:
            agg_df = pd.DataFrame({
                state: child_agg 
            })
            last_date = full_df['Date'].iloc[-1].to_pydatetime()
            start_date = last_date - dt.timedelta(days=agg_days)
            agg_df['Date'] = pd.to_datetime([(start_date + dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(agg_df))])
            if agg_df[state].max() > 10000:
                state_ax = plot_linechart(agg_df, 'Date', state, ax=state_ax, title='Statewise daily cases', show_peak=False)
    # plot aggregate chart for all states
    if plot:
        agg_df = pd.DataFrame({
            'states_agg': states_agg 
        })
        last_date = full_df['Date'].iloc[-1].to_pydatetime()
        start_date = last_date - dt.timedelta(days=agg_days)
        agg_df['Date'] = pd.to_datetime([(start_date + dt.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(agg_df))])
        plot_linechart(agg_df, 'Date', 'states_agg', title='Aggregate daily cases')

    return api


def export_tracker(api, fn="predictions.json"):
    # remove child
    tracker = {}
    for state in api:
        tracker[state] = {}
        for child in api[state]:
            for date in api[state][child]:
                tracker[state][date] = api[state][child][date]

    # aggregate predictions
    tracker['TT'] = {}
    for state in tracker:
        if state == 'TT':
            continue
        for date in tracker[state]:
            tracker['TT'][date] = tracker['TT'].get(date, {'delta':{}, 'total':{}})
            for k in ['delta']: #'total'
                tracker['TT'][date][k]['confirmed'] = tracker['TT'][date][k].get('confirmed', 0) + tracker[state][date][k]['confirmed']
                tracker['TT'][date][k]['deceased'] = tracker['TT'][date][k].get('deceased', 0) + tracker[state][date][k]['deceased']
                tracker['TT'][date][k]['recovered'] = tracker['TT'][date][k].get('recovered', 0) + tracker[state][date][k]['recovered']
                tracker['TT'][date][k]['active'] = tracker['TT'][date][k].get('active', 0) + tracker[state][date][k]['active']

    # export
    with open(fn, "w") as f:
        f.write(json.dumps(tracker, sort_keys=True))

        
def _aggregate_api(api):
    # remove child
    tracker = {}
    for state in api:
        tracker[state] = {}
        for child in api[state]:
            for date in api[state][child]:
                tracker[state][date] = api[state][child][date]
    
    # aggregate predictions
    tracker['TT'] = {}
    for state in tracker:
        if state == 'TT':
            continue
        for date in tracker[state]:
            tracker['TT'][date] = tracker['TT'].get(date, {})
            tracker['TT'][date]['c'] = tracker['TT'][date].get('c', 0) + tracker[state][date]['delta']['confirmed']
            tracker['TT'][date]['d'] = tracker['TT'][date].get('d', 0) + tracker[state][date]['delta']['deceased']
            tracker['TT'][date]['r'] = tracker['TT'][date].get('r', 0) + tracker[state][date]['delta']['recovered']
            tracker['TT'][date]['a'] = tracker['TT'][date].get('a', 0) + tracker[state][date]['delta']['active']
    return tracker


def export_videoplayer(api, prediction_date, fn=""):
    api = _aggregate_api(api)
    # read previous and export
    try:
        with open(fn, "r") as f:
            out = json.loads(f.read())
    except Exception as e:
        out = {}

    with open(fn, "w") as f:
        out[prediction_date] = {'TT': api['TT']}
        f.write(json.dumps(out, sort_keys=True))


def export_csv(api, prediction_date, fn=""):
    api = _aggregate_api(api)
    
    df_csv = pd.DataFrame(api['TT']).transpose()
    df_csv.drop(columns=['d', 'r', 'a'], inplace=True)
    df_csv.rename(columns={'c': prediction_date}, inplace=True)
    
    # read previous and export
    try:
        out = pd.read_csv(fn, index_col='date')
        df_csv = out.join(df_csv).fillna(0)
        df_csv[prediction_date] = df_csv[prediction_date].astype(int)
    except Exception as e:
        pass
    finally:
        df_csv.to_csv(fn, index_label='date')
