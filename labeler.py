import numpy as np
import pandas as pd
import logging
import os


class BinaryWinFinder:
    def __init__(self, df, symbol, long_or_short, candle_span, distance_threshold):
        self.df = df
        self.symbol =  f"({symbol.split("-")[0]})"
        self.long_or_short = long_or_short
        self.candle_span = candle_span
        self.distance_threshold = distance_threshold

    def _return_index_of_take_profit(self, index):
        df_len = len(self.df)
        if index + self.candle_span >= df_len:
            return np.Inf
        
        upper_stop = self.df.at[index, "upper_stop"]
        lower_stop = self.df.at[index, "lower_stop"]
        frame_segment = self.df.iloc[index+1:index+self.candle_span, :]
        upper_triggers = frame_segment.index[frame_segment[f'high {self.symbol}'] >= upper_stop].tolist()
        lower_triggers = frame_segment.index[frame_segment[f'low {self.symbol}'] <= lower_stop].tolist()
        
        if not (lower_triggers + upper_triggers):
            return np.Inf
        
        if lower_triggers and not upper_triggers:
            if self.long_or_short == "short":
                return min(lower_triggers)
            else:
                return np.Inf
            
        if not lower_triggers and upper_triggers:
            if self.long_or_short == "long":
                return min(upper_triggers)
            else: 
                return np.Inf
        
        if lower_triggers and upper_triggers:
            if self.long_or_short == "long":
                if min(lower_triggers) > min(upper_triggers):
                    return min(upper_triggers)
                else:
                    return np.Inf
                
            if self.long_or_short == "short":
                if min(lower_triggers) < min(upper_triggers):
                    return min(lower_triggers)
                else: 
                    return np.Inf
                
    def find_wins(self):
        
        self.df = self.df.reset_index(drop=True)
        self.df['upper_stop'] = self.df[f'close {self.symbol}'] * (1 + self.distance_threshold)
        self.df['lower_stop'] = self.df[f'close {self.symbol}'] * (1 - self.distance_threshold)
        self.df['took_profit_index'] = self.df.index.map(self._return_index_of_take_profit)
        self.df['win'] = np.where(self.df['took_profit_index'] != np.Inf, 1, 0)
        self.df = self.df.drop(['upper_stop', 'lower_stop', 'took_profit_index'], axis=1)
        self.df = self.df.reset_index(drop=True)
        self.df.to_csv('find_wins_test.csv')
            
        return self.df['win']