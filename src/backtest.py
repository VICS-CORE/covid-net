import datetime as dt
import numpy as np
import pandas as pd
import torch

from . import data
from matplotlib.dates import DayLocator, AutoDateLocator, ConciseDateFormatter

DEVICE='cpu'


def countrywise(model, cp, df, country='India', plot=True):
    c = country
    pop_fct = df.loc[df.location==c, 'population'].iloc[0] / 1000
    ip_aux = torch.tensor(
        np.array(df.loc[df.location==c, cp['config']['DS']['AUX_FEATURES']].iloc[0])[cp['config']['AUX_FEATURES']],
        dtype=torch.float32
    ).to(DEVICE)

    IP_SEQ_LEN = cp['config']['DS']['IP_SEQ_LEN']
    OP_SEQ_LEN = cp['config']['DS']['OP_SEQ_LEN']

    all_preds = []
    pred_vals = []
    real_vals = []

    test_data = np.array(df.loc[(df.location==c) & (df.total_cases>=100), cp['config']['DS']['FEATURES']].rolling(7, center=True, min_periods=1).mean() / pop_fct, dtype=np.float32)

    ip_shp = IP_SEQ_LEN, len(cp['config']['IP_FEATURES'])
    op_shp = OP_SEQ_LEN, len(cp['config']['OP_FEATURES'])

    for i in range(len(test_data) - IP_SEQ_LEN - OP_SEQ_LEN + 1):
        ip = torch.tensor(test_data[i : i+IP_SEQ_LEN, cp['config']['IP_FEATURES']])
        op = torch.tensor(test_data[i+IP_SEQ_LEN : i+IP_SEQ_LEN+OP_SEQ_LEN, cp['config']['OP_FEATURES']])
        ip = ip.to(DEVICE)
        op = op.to(DEVICE)

        pred = model.predict(
            ip.view(1, IP_SEQ_LEN, len(cp['config']['IP_FEATURES'])),
            aux_ip=ip_aux.view(1, len(cp['config']['AUX_FEATURES']))
        )

        all_preds.append(pred.view(op_shp).cpu().numpy() * pop_fct)
        pred_vals.append(pred.view(op_shp).cpu().numpy()[0] * pop_fct)
        real_vals.append(op.view(op_shp).cpu().numpy()[0] * pop_fct)

    # prepend first input
    nans = np.ndarray((IP_SEQ_LEN, len(cp['config']['OP_FEATURES'])))
    nans.fill(np.NaN)
    pred_vals = list(nans) + pred_vals
    real_vals = list(test_data[:IP_SEQ_LEN, cp['config']['OP_FEATURES']] * pop_fct) + real_vals
    # append last N-1 values
    nans = np.ndarray(op_shp)
    nans.fill(np.NaN)
    pred_vals.extend(nans[1:]) # pad with NaN
    real_vals.extend(op.view(op_shp).cpu().numpy()[1:] * pop_fct)

    real_vals = np.array(real_vals).reshape(len(test_data), len(cp['config']['OP_FEATURES']))
    pred_vals = np.array(pred_vals).reshape(len(test_data), len(cp['config']['OP_FEATURES']))

    accs = []
    for o in range(len(cp['config']['OP_FEATURES'])):
        cmp_df = pd.DataFrame({
            'actual': real_vals[:, o],
            'predicted0': pred_vals[:, o]
        })
        # set date
        start_date = df.loc[(df.location==c) & (df.total_cases>=100)]['date'].iloc[0]
        end_date = start_date + dt.timedelta(days=cmp_df.index[-1])
        cmp_df['Date'] = pd.Series([start_date + dt.timedelta(days=i) for i in range(len(cmp_df))])

        # plot noodles
        ax=None
        i=IP_SEQ_LEN
        mape=[]
        for pred in all_preds:
            cmp_df['predicted_cases'] = np.NaN
            cmp_df.loc[i:i+OP_SEQ_LEN-1, 'predicted_cases'] = pred[:, o]
            ape = np.array(100 * ((cmp_df['actual'] - cmp_df['predicted_cases']).abs() / cmp_df['actual']))
            mape.append(ape[~np.isnan(ape)])
            if plot: ax = cmp_df.plot(x='Date', y='predicted_cases', ax=ax, legend=False)
            i+=1
        
        total_acc = np.zeros(OP_SEQ_LEN)
        for m in mape: total_acc+=m
        elwise_mape = total_acc / len(mape)
        if plot: print("Day wise accuracy:", 100 - elwise_mape)
        acc = 100 - sum(elwise_mape)/len(elwise_mape)        
        accs.append(acc)
        acc_str = f"{acc:0.2f}%"

        # plot primary lines
        if plot:
            ax = cmp_df.plot(
                x='Date',
                y=['actual', 'predicted0'],
                figsize=(20,8),
                lw=5,
                title=c + ' | Daily predictions | ' + acc_str,
                ax=ax
            )
            mn_l = DayLocator()
            ax.xaxis.set_minor_locator(mn_l)
            mj_l = AutoDateLocator()
            mj_f = ConciseDateFormatter(mj_l, show_offset=False)
            ax.xaxis.set_major_formatter(mj_f)
    return accs


def statewise(model, cp, df, INFO, plot=True):
#     cp['config']['OP_FEATURES'] = [0] #fix for bug in 1.1740
    
    IP_SEQ_LEN = cp['config']['DS']['IP_SEQ_LEN']
    OP_SEQ_LEN = cp['config']['DS']['OP_SEQ_LEN']
    
    first_case_date = df['date'].min()
    
    accs = {}
    for state in INFO:
        accs[state] = {}
        for child in INFO[state]:
            pop_fct = child["popn"] / 1000
            ip_aux = torch.tensor(
                np.array(child["population_density"]),
                dtype=torch.float32
            ).to(DEVICE) if "population_density" in child else None
            
            child_df = df.loc[(df['state']==state) & (df['name']==child['name'])]
            child_df = data.prefill(data.expand(child_df), first_case_date)
            child_df['new_cases'] = child_df['confirmed'] - child_df['confirmed'].shift(1).fillna(0)
            child_df['new_deaths'] = child_df['deceased'] - child_df['deceased'].shift(1).fillna(0)
            child_df['new_recovered'] = child_df['recovered'] - child_df['recovered'].shift(1).fillna(0)
            child_df['new_tests'] = child_df['tested'] - child_df['tested'].shift(1).fillna(0)
            test_data = np.array(child_df[cp['config']['DS']['FEATURES']].rolling(7, center=True, min_periods=1).mean() / pop_fct, dtype=np.float32)
            
            all_preds = []
            pred_vals = []
            real_vals = []
            
            ip_shp = IP_SEQ_LEN, len(cp['config']['IP_FEATURES'])
            op_shp = OP_SEQ_LEN, len(cp['config']['OP_FEATURES'])
            
            for i in range(len(test_data) - IP_SEQ_LEN - OP_SEQ_LEN + 1):
                ip = torch.tensor(test_data[i : i+IP_SEQ_LEN, cp['config']['IP_FEATURES']])
                op = torch.tensor(test_data[i+IP_SEQ_LEN : i+IP_SEQ_LEN+OP_SEQ_LEN, cp['config']['OP_FEATURES']])
                ip = ip.to(DEVICE)
                op = op.to(DEVICE)
                
                args = [ip.view(1, IP_SEQ_LEN, len(cp['config']['IP_FEATURES']))]
                if len(cp['config'].get('AUX_FEATURES', [])):
                    args.append(ip_aux.view(1, len(cp['config']['AUX_FEATURES'])))
                pred = model.predict(*args)
                
                all_preds.append(pred.view(op_shp).cpu().numpy() * pop_fct)
                pred_vals.append(pred.view(op_shp).cpu().numpy()[0] * pop_fct)
                real_vals.append(op.view(op_shp).cpu().numpy()[0] * pop_fct)
            
            # prepend first input
            nans = np.ndarray((IP_SEQ_LEN, len(cp['config']['OP_FEATURES'])))
            nans.fill(np.NaN)
            pred_vals = list(nans) + pred_vals
            real_vals = list(test_data[:IP_SEQ_LEN, cp['config']['OP_FEATURES']] * pop_fct) + real_vals
            # append last N-1 values
            nans = np.ndarray(op_shp)
            nans.fill(np.NaN)
            pred_vals.extend(nans[1:]) # pad with NaN
            real_vals.extend(op.view(op_shp).cpu().numpy()[1:] * pop_fct)
            
            real_vals = np.array(real_vals).reshape(len(test_data), len(cp['config']['OP_FEATURES']))
            pred_vals = np.array(pred_vals).reshape(len(test_data), len(cp['config']['OP_FEATURES']))
            
            accs[state][child['name']] = []
            for o in range(len(cp['config']['OP_FEATURES'])):
                cmp_df = pd.DataFrame({
                    'actual': real_vals[:, o],
                    'predicted0': pred_vals[:, o]
                })
                # set date
                cmp_df['Date'] = pd.Series([first_case_date + dt.timedelta(days=i) for i in range(len(cmp_df))])

                # plot noodles
                ax=None
                i=IP_SEQ_LEN
                mape=[]
                for pred in all_preds:
                    # skip noodles if cases < N on any day
                    if any(cmp_df.loc[i:i+OP_SEQ_LEN-1, 'actual'] < 100):
                        i+=1
                        continue
                    cmp_df['predicted_cases'] = np.NaN
                    cmp_df.loc[i:i+OP_SEQ_LEN-1, 'predicted_cases'] = pred[:, o]
                    ape = np.array(100 * ((cmp_df['actual'] - cmp_df['predicted_cases']).abs() / (cmp_df['actual'].abs() + 1)))
                    ape = ape[~np.isnan(ape)] # remove nans
                    if len(ape): mape.append(ape)
                    if plot: ax = cmp_df.plot(x='Date', y='predicted_cases', ax=ax, legend=False)
                    i+=1
                total_acc = np.zeros(OP_SEQ_LEN)
                for m in mape: total_acc+=m
                elwise_mape = total_acc / len(mape)
                acc = 100 - sum(elwise_mape)/len(elwise_mape)
                acc = np.nan if acc < 0 else acc
                accs[state][child['name']].append(acc)
                acc_str = f"{acc:0.2f}%"
                if plot: 
                    print(state, child['name'], acc_str)
                    print("daywise accuracy:", 100 - elwise_mape)

                # plot primary lines
                if plot:
                    ax = cmp_df.plot(
                        x='Date',
                        y=['actual', 'predicted0'],
                        figsize=(10,5),
                        lw=5,
                        title=child['name'] + ' | Daily predictions | ' + acc_str,
                        ax=ax
                    )
                    mn_l = DayLocator()
                    ax.xaxis.set_minor_locator(mn_l)
                    mj_l = AutoDateLocator()
                    mj_f = ConciseDateFormatter(mj_l, show_offset=False)
                    ax.xaxis.set_major_formatter(mj_f)
    
    # agg accs
    state_accs = []
    weights = []
    for state in accs:
        child_accs = []
        for child in accs[state]:
            child_accs.append(accs[state][child])
            weights.append(INFO[state][0]['popn']) # dirty hack
        a = np.nanmean(np.array(child_accs))
        print(state, a)
        state_accs.append(a)

    # simple average
#     a = np.nanmean(np.array(state_accs))
    # weighted avg
    wts = np.array(weights)
    arr = np.array(state_accs)
    indices = ~np.isnan(arr)
    a = np.average(arr[indices], weights=wts[indices])

    print("India", a)
    return a
