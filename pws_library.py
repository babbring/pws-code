import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt

def get_data(tickers, start, end):  
    prices_df = pd.DataFrame()
    columns_names = []

    for ticker in tickers:
        df = yf.download(ticker, auto_adjust=False, start=start, end=end)
        new_column = df['Close']
        prices_df = pd.concat([prices_df, new_column], axis=1)
        columns_names.append(ticker)
    prices_df.columns = columns_names
    return prices_df


        
class Backtest:
    
    def __init__(self, instrument, futures, df, rollovers, jaren):
        self.instrument = instrument
        self.futures = futures
        self.df = df
        self.rollovers = rollovers
        self.years = jaren

        if self.df.empty:
            print("geen data geleverd")
        else:
            self.bereken_zscore()
            self.genereer_signalen()
            self.koppel_signalen()
            self.bereken_winst()
            self.geef_resultaten()
        
    def bereken_zscore(self):
        lookback = 20

        self.df['spread'] = self.df[self.instrument] - self.df[self.futures]
        self.df['zscore'] = (self.df['spread'] - self.df['spread'].rolling(lookback).mean()) / self.df['spread'].rolling(lookback).std()
        self.df.dropna(inplace=True)
    
    def genereer_signalen(self):
        entryZscore = 2
        exitZscore = 0
        
        conditions = [(self.df.zscore < -entryZscore), (self.df.zscore >= -exitZscore)]
        choices = [1, 0]
        self.df['num_long'] = np.select(conditions, choices, default=np.NaN)
        
        conditions1 = [(self.df.zscore > entryZscore), (self.df.zscore <= exitZscore)]
        choices1 = [-1, 0]
        self.df['num_short'] = np.select(conditions1, choices1, default=np.NaN)
    
    
    def koppel_signalen(self):
        self.df['num_short'] = self.df['num_short'].fillna(method='ffill', inplace=False).fillna(value=0, inplace=False)
        self.df['num_long'] = self.df['num_long'].fillna(method='ffill', inplace=False).fillna(value=0, inplace=False)
        self.df['num_units'] = self.df['num_long'] + self.df['num_short']
        
        self.trades=0
        long_pos=False
        short_pos=False
        for index, row in self.df.iterrows():
            if not long_pos and row['num_units'] == 1:
                long_pos = True
                short_pos = False
                self.trades += 1
            if not short_pos and row['num_units'] == -1:
                short_pos = True
                long_pos = False
                self.trades += 1
            if long_pos or short_pos and row['num_units'] == 0:
                short_pos= False
                long_pos= False

        self.df[f"{self.instrument}_pos"] = self.df['num_units'] * self.df[self.instrument]
        self.df[f"{self.futures}_pos"] = -self.df['num_units'] * self.df[self.futures]
        
        
    def bereken_winst(self):
        self.df[f'{self.instrument}_pnl'] = (self.df[self.instrument]-self.df[self.instrument].shift())/self.df[self.instrument].shift()*self.df[f'{self.instrument}_pos'].shift()
        self.df[f'{self.futures}_pnl'] = (self.df[self.futures]-self.df[self.futures].shift())/self.df[self.futures].shift()*self.df[f'{self.futures}_pos'].shift()
        self.df['pnl'] = self.df[f'{self.instrument}_pnl'] + self.df[f'{self.futures}_pnl']
        
        mask = []
        for value in self.df.index.tolist():
            mask.append(value in self.rollovers)
        self.df.iloc[mask]['pnl'] = 0
        
        
        self.ret = self.df['pnl'] / (np.abs(self.df[f'{self.instrument}_pos'].shift()) + np.abs(self.df[f'{self.futures}_pos'].shift()))
        self.ret = self.ret.fillna(0)
        
        self.returns_df = pd.DataFrame(self.ret, columns=['ret'])
        
        cum_ret = []
        for i in range(len(self.ret)):
            if i == 0:
                cum_ret.append(self.ret.iloc[i])
            else:
                cum_ret.append((self.ret.iloc[i] + 1) * (cum_ret[i-1]+1) - 1)
        

        self.returns_df['cum_ret'] = cum_ret
        
        self.sharpe = (self.returns_df['ret'].mean() - (1.02570**(1/365)-1))/self.returns_df['ret'].std()
        
        self.max_drawdown = 0
        for i in range(len(self.returns_df['cum_ret'])):
            current_cumret = self.returns_df.iloc[i]['cum_ret']
            dd = (current_cumret - self.returns_df.iloc[:i+1]['cum_ret'].max())/(self.returns_df.iloc[:i+1]['cum_ret'].max()+1)
            if dd < self.max_drawdown:
                self.max_drawdown = dd

    def geef_resultaten(self):
        (self.returns_df['cum_ret']*100).plot(figsize=(14,6))
        print(f"totale rendement: {np.round(self.returns_df.iloc[-1]['cum_ret']*100, decimals=2)}%")
        print(f"jaarlijkse rendement: {np.round((1+self.returns_df.iloc[-1]['cum_ret'])**(1/self.years)*100-100, decimals=2)}%")
        print(f'sharpe-ratio: {np.round(self.sharpe, decimals=2)}')
        print(f'maximale tijdelijke verlies: {np.round(self.max_drawdown*100, decimals=2)}%')