import numpy as np
import pandas as pd
import logging
import os


class BinaryWinFinder:
    def __init__(self, df, long_or_short, candle_span, distance_threshold):
        self.df = df
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
        upper_triggers = frame_segment.index[frame_segment['high'] >= upper_stop].tolist()
        lower_triggers = frame_segment.index[frame_segment['low'] <= lower_stop].tolist()
        
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
        try:
            self.df = self.df.reset_index(drop=True)
            self.df['upper_stop'] = self.df['close'] * (1 + self.distance_threshold)
            self.df['lower_stop'] = self.df['close'] * (1 - self.distance_threshold)
            self.df['took_profit_index'] = self.df.index.map(self._return_index_of_take_profit)
            self.df['win'] = np.where(self.df['took_profit_index'] != np.Inf, 1, 0)
            self.df = self.df.drop(['upper_stop', 'lower_stop', 'took_profit_index'], axis=1)
            self.df = self.df.reset_index(drop=True)

        except Exception as e:
            # Log the error with the relevant data
            logging.error("Error during timestamp conversion and formatting: %s", e)
            
            # Log the DataFrame that caused the error
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            error_log_filename = f'error_data_{timestamp}.csv'
            log_dir = os.makedirs("logs", exist_ok=True)
            self.df.to_csv(log_dir + "/" + error_log_filename, index=False)
            logging.error("Data causing error saved to %s", error_log_filename)
            
        return self.df['win']