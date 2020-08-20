import datetime as dt
import numpy as np
import pandas as pd
import torch

from matplotlib.dates import DayLocator, AutoDateLocator, ConciseDateFormatter

DEVICE='cpu'


def classic(model, cp, df, country='India', plot=True):
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