import os
import time
import maps
import numpy as np
import pandas as pd
from datetime import datetime
from kucoin_api import KucoinApi  # Make sure this module is correctly installed and imported
from format import format_time
from decorators import time_it


class DataGetter:
    """A class to fetch and process cryptocurrency data from the Kucoin API.

    Attributes:
    -----------
    symbol : str
        The trading pair symbol (e.g., "ETH-USDT").
    timeframe : str
        The timeframe for fetching data (e.g., "5min").
    """

    def __init__(self, symbol, timeframe):
        """Initializes the DataGetter with a symbol and timeframe.

        Parameters:
        -----------
        symbol : str
            The trading pair symbol (e.g., "ETH-USDT").
        timeframe : str
            The timeframe for fetching data (e.g., "5min").
        """
        self.api = KucoinApi()
        self.symbol = symbol
        self.timeframe = timeframe
        self.timeframe_in_minutes = maps.timeframes_in_minutes[timeframe]
        if self.timeframe_in_minutes is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
    def generate_start_and_end(self):
        """Generates start and end Unix timestamps for the data fetching window.

        Returns:
        --------
        tuple
            A tuple containing the start and end timestamps as strings.
        """
        now = datetime.now()
        now_as_unixtimestamp = int(time.mktime(now.timetuple()))
        timeframe_in_seconds = self.timeframe_in_minutes * 60
        segment_length = timeframe_in_seconds * 1500
        start = str(now_as_unixtimestamp - segment_length)
        end = str(now_as_unixtimestamp)
        return start, end
    
    def get_candles(self, start_at=None, end_at=None, delay=0):
        """Fetches candlestick data from the Kucoin API.

        Parameters:
        -----------
        start_at : str, optional
            The start timestamp for fetching data.
        end_at : str, optional
            The end timestamp for fetching data.
        delay : int, optional
            The delay in seconds before making the request (default is 0).

        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing the candlestick data.
        """
        time.sleep(delay)
        if start_at is None and end_at is None:
            start_at, end_at = self.generate_start_and_end()
        payload = {
            "symbol": self.symbol, 
            "startAt": start_at,
            "endAt": end_at, 
            "type": self.timeframe
        }
        try:
            response = self.api.process_request(request_type="GET", endpoint=self.api.endpoints['candles'], **payload)
            candles = response['data']
        except KeyError:
            print("Retrying get_candles response due to KeyError")
            time.sleep(1)
            return self.get_candles(start_at, end_at, delay)

        candles = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "close", "high", "low", "volume", "amount"]
        ).astype(float)
        candles['timestamp'] = candles['timestamp'].astype(int)
        check = self.check_timestamp_sequence(candles)
        print(f"""
              FETCHING params: {self.symbol}, 
              {self.timeframe}, 
              start: {start_at}, 
              end: {end_at}
              """)
        print("Received Dataframe: ", check)
        return candles  
    
    def check_timestamp_sequence(self, df):
        """Checks the timestamp sequence for gaps and logs any issues.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to check for timestamp sequence.

        Returns:
        --------
        dict
            A dictionary with details about the timestamp sequence.
        """
        res = {
            "first_timestamp": 0,
            "last_timestamp": 0,
            "gaps": 0,
            "count": df['timestamp'].size,
        }
        diff = df['timestamp'].astype(int).diff().dropna().reset_index(drop=True) * -1
        gaps = np.where(diff != self.timeframe_in_minutes * 60, 1, 0)
        res['gaps'] = any(gaps)
        res['first_timestamp'] = df['timestamp'][0]
        res['last_timestamp'] = df['timestamp'][df['timestamp'].size - 1]
        if res["gaps"]: 
            timestamp = format_time(datetime.now(), 'timestamp')
            os.makedirs("data_to_inspect", exist_ok=True)
            df.to_csv(f"data_to_inspect/{timestamp}.csv")
            print(f"""
                GAP: dataframe saved to data_to_inspect as {timestamp}
                  """)
            raise Exception("GAP detected")
        return res
    
    @time_it
    def get_mass_candles(self, span, delay=0):
        """Fetches a large amount of candlestick data in segments.

        Parameters:
        -----------
        span : int
            The total time span for fetching data.
        delay : int, optional
            The delay in seconds before making each request (default is 0).

        Returns:
        --------
        pandas.DataFrame
            A DataFrame containing the concatenated candlestick data.
        """
        most_recent_timestamp = self.get_candles(delay=delay)['timestamp'][0]
        segment_len = self.timeframe_in_minutes * 60 * 1000
        num_of_segments = span // 1000
        end = most_recent_timestamp
        segment_ends = {}
        for _ in range(num_of_segments):
            segment_ends[end] = int(end - segment_len)
            end = int(end - segment_len)
        
        data = [self.get_candles(start_at=str(y), end_at=str(x)) for x, y in segment_ends.items()]
        df = pd.concat(data).reset_index(drop=True)
        self.check_timestamp_sequence(df)
        return df