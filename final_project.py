#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 01:38:56 2017

@author: himanshu
"""

import quandl
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as stats
import numpy as np
import matplotlib.pyplot as plt

# Start and End Dates
start = '2014-01-01'    
end = '2016-01-01'

# Imports Continuous Contracts from Quandl
def import_data():
    # MCX Gold Contract 
    gc = quandl.get("CHRIS/MCX_GC1", authtoken="KfYDcdPudPzu6X2si99D", start_date=start, end_date=end)
    # MCX Silver Contract
    si = quandl.get("CHRIS/MCX_SI1", authtoken="KfYDcdPudPzu6X2si99D", start_date=start , end_date=end)
    # MCX Aluminium Contract
    al = quandl.get("CHRIS/MCX_AL1", authtoken="KfYDcdPudPzu6X2si99D", start_date=start , end_date=end)
    # MCX Lead Contract
    pb = quandl.get("CHRIS/MCX_PB1", authtoken="KfYDcdPudPzu6X2si99D", start_date=start , end_date=end)
    # MCX Nickel Contract
    ni = quandl.get("CHRIS/MCX_NI1", authtoken="KfYDcdPudPzu6X2si99D", start_date=start , end_date=end)
    # MCX Zinc Contract
    zn = quandl.get("CHRIS/MCX_ZN1", authtoken="KfYDcdPudPzu6X2si99D", start_date=start , end_date=end)
    # MCX Copper Contract
    cu = quandl.get("CHRIS/MCX_CU1", authtoken="KfYDcdPudPzu6X2si99D", start_date=start , end_date=end)
    # MCX Crude Oil Contract
    cl = quandl.get("CHRIS/MCX_CL1", authtoken="KfYDcdPudPzu6X2si99D", start_date=start , end_date=end)
    # MCX Natural Gas Contract
    ng = quandl.get("CHRIS/MCX_NG1", authtoken="KfYDcdPudPzu6X2si99D", start_date=start , end_date=end)
    # MCX Cardamom Contract
    crdm = quandl.get("CHRIS/MCX_CRDM1", authtoken="KfYDcdPudPzu6X2si99D", start_date=start , end_date=end)
    # MCX Cotton Contract
    ct = quandl.get("CHRIS/MCX_CT1", authtoken="KfYDcdPudPzu6X2si99D", start_date=start , end_date=end)

    # List containing commodity data
    data = [gc, si, al, pb, ni, zn, cu, cl, ng, crdm, ct]
    # List containing commodity symbols
    symbols=['GC', 'SI', 'AL', 'PB', 'NI', 'ZN', 'CU', 'CL', 'NG', 'CRDM', 'CT']
    
    return data, symbols

# Finds pairs from the imported data
def find_pairs(data, symbols):
    
    lookback = 360
    
    # Initialize an empty list of pairs
    pairs_sym = []
    pairs_df = []
    
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            df = pd.concat([data[i].Close, data[j].Close], axis=1)
            df = df.fillna(method='ffill')
            df = df.dropna()
            df.columns=['Y', 'X']
            
            # Get days on which ADF test confirms stationarity 
            # and the days on which the pair is co-integrated
            # coint: contains days on which pair is co-integrated.
            # st: contains days on ADF test confirms stationarity for the pair.
            coint, st = get_coint(df, lookback)
            
            # Calculate percentage of the days.
            # If percentage is more than 40%, add it to list of pairs
            try:
                days = float(len(coint.dropna()))/len(st)
                res = np.corrcoef(df.Y, df.X)
                if days > 0.4 and res[0,1] > 0.6:
                    print symbols[i], symbols[j], days, res[0,1]
                    pairs_sym.append((symbols[i], symbols[j], days, res[0,1]))
                    pairs_df.append(df)
            except:
                continue
    return pairs_df, pairs_sym

# Finds days on which ADF test and co-integration test both pass for a pair 
def get_coint(df, lookback):
    # Find rolling spreads with a window = lookback
    spreads = {}
    for i in range(lookback, len(df)):
        x_cons = sm.add_constant(df.X.iloc[i-lookback:i])
        res = sm.OLS(df.Y.iloc[i-lookback:i], x_cons).fit()
        spread = df.Y.iloc[i-lookback:i]-res.params[1]*df.X.iloc[i-lookback:i]
        spreads[df.index[i]] = spreads.get(df.index[i], spread)
    
    # Find days on which ADF test confirms stationarity
    st = pd.Series(index=df.index)
    for key, spread in spreads.iteritems():
        r = stats.adfuller(spread, maxlag=1)
        if r[1] < 0.05:
            st.ix[key] = r
    st = st.dropna()
    
    # Out of days on which ADF test confirms stationarity,
    # find the days on which co-integration is confirmed
    coint = pd.Series(index=st.index)
    for i in st.index:
        loc = df.index.get_loc(i)
        r = stats.coint(df.Y.iloc[loc-lookback:loc], df.X.iloc[loc-lookback:loc])
        if r[1] < 0.05:
            coint.ix[i] = r[1]
    coint = coint.dropna()
    # coint: contains days on which pair is co-integrated.
    # st: contains days on ADF test confirms stationarity for the pair.
    return coint, st

# Backtests a strategy. Generates Z-Scores, Trade Signal and Returns.
def backtest(df, coint=None, strategy=1):
    # Find the price ratio between the pair
    ratio = np.log(df.Y/df.X)
    ratio.name = 'Ratio'
    
    # Calculate Simple Moving Average and
    # Moving Standard Deviation of the price ratio
    mavg = ratio.rolling(window = 30).mean()
    mavg.name = 'MA_30'
    std = ratio.rolling(window = 30).std()
    std.name = 'STD_30'
    
    # Calculate Z-Score
    z_score = (ratio-mavg)/std
    z_score.name = 'Z'
              
    df_1 = pd.concat([df, ratio, mavg, std, z_score], axis=1)
    df_1 = df_1.dropna()
    
    # Find log returns for each commodity in the pair
    Y_log = np.log(df_1.Y/df.Y.shift(1))
    Y_log.name = 'Y_log'
    X_log = np.log(df_1.X/df.X.shift(1))
    X_log.name = 'X_log'
    
    df_1 = pd.concat([df_1, Y_log, X_log], axis=1)
    df_1 = df_1.dropna()
    
    # Z-Score thresholds
    entryZ = 2.0
    exitZ = 0.0
    
    inLong = None
    inShort = None
    
    Y_sign = pd.Series(index=df_1.index)
    Y_sign.name = 'Y_sign'
    X_sign = pd.Series(index=df_1.index)
    X_sign.name = 'X_sign'
    
    rets = pd.Series(index=df_1.index)
    rets.name = 'returns'
    
    # Generate Trade signals and returns for Strategy 1 
    if coint is None:
        for i in range(len(df_1)):
            if (inLong == None) and (df_1.Z.iloc[i] < -entryZ):
                Y_sign.iloc[i] = 1
                X_sign.iloc[i] = -1
                inLong = i
                inShort = None
    
            if (inLong != None) and (df_1.Z.iloc[i] >= -exitZ):
                Y_sign.iloc[i] = -1
                X_sign.iloc[i] = 1
                rets_y = np.sum(df_1.Y_log.iloc[inLong+1:i+1])*Y_sign.iloc[inLong]
                rets_x = np.sum(df_1.X_log.iloc[inLong+1:i+1])*X_sign.iloc[inLong]
                rets.iloc[i] = (rets_y+rets_x)/2
                inLong = None
                inShort = None
                
            if (inShort == None) and (df_1.Z.iloc[i] > entryZ):
                Y_sign.iloc[i] = -1
                X_sign.iloc[i] = 1
                inLong = None
                inShort = i
        
            if (inShort != None) and (df_1.Z.iloc[i] <= exitZ):
                Y_sign.iloc[i] = 1
                X_sign.iloc[i] = -1
                rets_y = np.sum(df_1.Y_log.iloc[inShort+1:i+1])*Y_sign.iloc[inShort]
                rets_x = np.sum(df_1.X_log.iloc[inShort+1:i+1])*X_sign.iloc[inShort]
                rets.iloc[i] = (rets_y+rets_x)/2
                inLong = None
                inShort = None
    
    # Generate Trade signals and returns for Strategy 1     
    elif coint is not None and strategy == 2:
        for i in range(len(df_1)):
            index = df_1.index[i]
            if index in coint.index and (inLong == None) and (df_1.Z.iloc[i] < -entryZ):
                Y_sign.iloc[i] = 1
                X_sign.iloc[i] = -1
                inLong = i
                inShort = None
    
            if index in coint.index and (inLong != None) and (df_1.Z.iloc[i] >= -exitZ):
                Y_sign.iloc[i] = -1
                X_sign.iloc[i] = 1
                rets_y = np.sum(df_1.Y_log.iloc[inLong+1:i+1])*Y_sign.iloc[inLong]
                rets_x = np.sum(df_1.X_log.iloc[inLong+1:i+1])*X_sign.iloc[inLong]
                rets.iloc[i] = (rets_y+rets_x)/2
                inLong = None
                inShort = None
                
            if index in coint.index and (inShort == None) and (df_1.Z.iloc[i] > entryZ):
                Y_sign.iloc[i] = -1
                X_sign.iloc[i] = 1
                inLong = None
                inShort = i
        
            if index in coint.index and (inShort != None) and (df_1.Z.iloc[i] <= exitZ):
                Y_sign.iloc[i] = 1
                X_sign.iloc[i] = -1
                rets_y = np.sum(df_1.Y_log.iloc[inShort+1:i+1])*Y_sign.iloc[inShort]
                rets_x = np.sum(df_1.X_log.iloc[inShort+1:i+1])*X_sign.iloc[inShort]
                rets.iloc[i] = (rets_y+rets_x)/2
                inLong = None
                inShort = None
            
    df_1 = pd.concat([df_1, Y_sign, X_sign, rets], axis=1)
    return df_1

# Backtest on the basis of strategy selected
def backtest_strategy(pairs_df, pairs_sym, strategy=1):        
    rets = None
    for i in range(len(pairs_df)):
        df = pairs_df[i]
        df_1 = None
        if strategy == 1:
            df_1 = backtest(df)
        elif strategy == 2:
            coint,st = get_coint(df, lookback=360)
            df_1 = backtest(df, coint, strategy=2)
        
        ret = df_1.returns
        ret.name = pairs_sym[i][0]+'_'+pairs_sym[i][1]
        if rets is None:
            rets = ret
        else:
            rets = pd.concat([rets, ret], axis=1)
    return rets

# Performs out of sample backtesting of pairs
def out_sample_test(pairs, start_dt, end_dt, strategy=1):
    rets = None
    for p in pairs:
        y = quandl.get('CHRIS/MCX_'+p[0]+'1', authtoken="KfYDcdPudPzu6X2si99D", start_date=start_dt, end_date=end_dt)
        x = quandl.get('CHRIS/MCX_'+p[1]+'1', authtoken="KfYDcdPudPzu6X2si99D", start_date=start_dt, end_date=end_dt)
        
        df = pd.concat([y.Close, x.Close], axis=1)
        df = df.fillna(method='ffill')
        df = df.dropna()
        df.columns=['Y', 'X']
    
        df_1 = None
        if strategy == 1:
            df_1 = backtest(df)
        elif strategy == 2:
            coint,st = get_coint(df, lookback=360)
            df_1 = backtest(df, coint, strategy=2)
        
        ret = df_1.returns
        ret.name = p[0]+'_'+p[1]
        if rets is None:
            rets = ret
        else:
            rets = pd.concat([rets, ret], axis=1)
    return rets

# Calculates performance parameters and generates equity curve
def calc_performance(rets, num_of_pairs, lev=1):
    # Combine daily returns of all pairs to generate 
    # daily returns of the portfolio
    total_rets = pd.Series(index=rets.index)
    total_rets.name = 'Total Returns'
    for idx in total_rets.index:
        total_rets.ix[idx] = np.sum(rets.ix[idx])
    
    # Generate Equity Curve
    eq_curve = pd.Series(index=total_rets.index)
    for i in range(len(total_rets)):
        if i > 0:
            if total_rets.iloc[i]==0: 
                eq_curve.iloc[i] = eq_curve.iloc[i-1]
            else:
                eq_curve.iloc[i] = eq_curve.iloc[i-1]*(1+lev*total_rets.iloc[i])
        else:
            # Initially, an investment of Rs. 100
            eq_curve.iloc[i]=100.0

    
    ## Calculate various performance parameters
    # CAGR
    cagr = ((eq_curve.iloc[-1]/eq_curve.iloc[0])**(1.0/2)-1)*100
    # Positive Trades
    pos_trades = [total_rets.ix[idx] for idx in total_rets.index if total_rets.ix[idx] > 0]
    # Negative Trades
    neg_trades = [total_rets.ix[idx] for idx in total_rets.index if total_rets.ix[idx] < 0]
    # Number of Positive Trades
    num_profit_trades = len(pos_trades)
    # Number of Negative Trades
    num_loss_trades = len(neg_trades)
    # Hit Ratio
    hit_ratio = 0
    if num_profit_trades > 0:
        hit_ratio = float(num_profit_trades)/(num_profit_trades+num_loss_trades)
    # Average Profit
    avg_profit = np.mean(pos_trades)
    
    # Sharpe Ratio
    sharpe = 0.0
    if not all(total_rets[0] == item for item in total_rets):
        sharpe = np.sqrt(252)*(np.mean(total_rets)/np.std(total_rets))
    
    # Maximum Drawdown
    prev_max = 0
    prev_min = 0
    max_loc = 0
    for i in range(1, len(eq_curve)):
        if eq_curve.iloc[i] > eq_curve.iloc[max_loc]:
            max_loc = i
        elif (eq_curve.iloc[max_loc]-eq_curve.iloc[i]) > (eq_curve.iloc[prev_max]-eq_curve.iloc[prev_min]):
            prev_max = max_loc
            prev_min = i
    max_dd = (eq_curve.iloc[prev_min]/eq_curve.iloc[prev_max])-1
    
    l = ['CAGR', 'Sharpe-Ratio', 'Positive Trades', 'Negative Trades', 'Hit-Ratio', 'Average Profit']
    v = [cagr, sharpe, num_profit_trades, num_loss_trades, hit_ratio*100, avg_profit*100]
    if hit_ratio < 1.0:
        # Average Loss
        avg_loss = np.mean(neg_trades)
        l.append('Average Loss')
        v.append(avg_loss*100)
    l.append('Maximum Drawdown')
    v.append(max_dd*100)
    labels = pd.Series(l)
    vals = pd.Series(v)
    
    df_params = pd.concat([labels, vals], axis=1)
    df_params.columns = ['Parameter', 'Value']
    df_params.index = range(1, len(df_params)+1)
    
    # Print performance parameters 
    print df_params
    
    # Plot Equity Curve along with maximum drawdown
    plt.plot(eq_curve.index, eq_curve, label='Equity')
    plt.title('Equity Curve')
    plt.xlabel('Year-Month')
    plt.ylabel('Equity')
    plt.legend(loc='upper left')
    
    if prev_min > prev_max:
        plt.scatter(eq_curve.index[prev_max],eq_curve.iloc[prev_max], color='red')
        plt.scatter(eq_curve.index[prev_min],eq_curve.iloc[prev_min], color='red')
        
    return eq_curve, total_rets


# Import data
data, symbols = import_data()

# Find pairs
pairs_df, pairs_sym = find_pairs(data, symbols)

#==============================================================================
# Uncomment the following two lines of code for in-sample Backtesting using 
# Strategy 1 
#==============================================================================
rets = backtest_strategy(pairs_df, pairs_sym, strategy = 1)
eq_curve, total_rets = calc_performance(rets, len(pairs_sym), lev=10)

#==============================================================================
# Uncomment the following two lines of code for in-sample Backtesting using 
# Strategy 2
#==============================================================================
#rets = backtest_strategy(pairs_df, pairs_sym, strategy = 2)
#eq_curve = calc_performance(rets, len(pairs_sym))


#==============================================================================
# Uncomment the following two lines of code for out of sample Backtesting using 
# Strategy 1 
#==============================================================================
#rets = out_sample_test(pairs_sym, start_dt='2016-01-02', end_dt='2017-01-01', strategy=1)
#eq_curve = calc_performance(rets, len(pairs_sym))


#==============================================================================
# Uncomment the following two lines of code for out of sample Backtesting using 
# Strategy 2 
#==============================================================================
#rets = out_sample_test(pairs_sym, start_dt='2016-01-02', end_dt='2017-01-01', strategy=2)
#eq_curve = calc_performance(rets, len(pairs_sym))


    

