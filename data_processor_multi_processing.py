import gc
from data_service import DataService
from labeler import BinaryWinFinder 
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import pandas as pd
import logging

logging.basicConfig(level=logging.ERROR)

def get_market_data_for_symbols(list_of_symbols, timeframe):
    """Queries all market data available in the database for the selected symbols and timeframe.

    Each series name includes the symbol.

    Params:
        list_of_symbols: format like 'ETH-USDT'
        timeframe: format like '1min'

    Returns: 
        merged dataframe
    """
    dataframes = []
    for symbol in list_of_symbols:
        df = DataService(symbol, timeframe).load_market_data()
        df.columns = [col + f" ({symbol.split('-')[0]})" if col != "timestamp" else col for col in df.columns]
        dataframes.append(df)

    # Merge all dataframes on the 'timestamp' column
    merged_df = dataframes[0]
    for i in range(1, len(dataframes)):
        merged_df = pd.merge(merged_df, dataframes[i], on='timestamp', how='inner')

    return merged_df

def extract_previous_candles_with_outcome(data, window_size=45):
    results = []
    for idx in range(window_size, len(data)):
        df = data.iloc[idx - window_size:idx].copy()
        win_value = data.iloc[idx]['win']
        timestamp_value = data.iloc[idx]['timestamp']
        results.append({
            'data': df, 
            'label': int(win_value), 
            'timestamp': int(timestamp_value)
        })
        # print(df, win_value, timestamp_value)
    return results

def extract_previous_candles_with_outcome_in_batches(data, window_size=45, batch_size=1000):
    """
    Extract previous candles with outcomes in batches.

    Parameters:
    data (pd.DataFrame): The market data with a 'win' column.
    window_size (int): The size of the window to consider for each sample.
    batch_size (int): The number of samples to process in each batch.

    Returns:
    list: A list of dictionaries with 'data', 'label', and 'timestamp' keys.
    """
    results = []
    num_batches = (len(data) - window_size) // batch_size + 1
    
    for batch in range(num_batches):
        start_idx = batch * batch_size + window_size
        end_idx = min((batch + 1) * batch_size + window_size, len(data))

        for idx in range(start_idx, end_idx):
            df = data.iloc[idx - window_size:idx].copy()
            win_value = data.iloc[idx]['win']
            timestamp_value = data.iloc[idx]['timestamp']
            results.append({
                'data': df, 
                'label': int(win_value), 
                'timestamp': int(timestamp_value)
            })
            # print(df, win_value, timestamp_value)
    
    return results

def drop_unwanted_columns(df):
    drop_cols = [col for col in df.columns 
                 if 'id (' in col 
                 or 'symbol (' in col 
                 or 'timeframe (' in col]
    df = df.drop(drop_cols, axis=1)
    return df

def label_wins(df, symbol, long_or_short, candle_span, pct_chg_threshold=0.01):
    win_finder = BinaryWinFinder(df, f"{symbol}", long_or_short, candle_span, pct_chg_threshold)
    return win_finder.find_wins()

def scale_dataframe(df, symbol_a, symbol_b):
    asset_a, asset_b = [_.split("-")[0] for _ in (symbol_a, symbol_b)]
    cols_to_scale = [
        f'open ({asset_b})', f'close ({asset_b})', f'high ({asset_b})', f'low ({asset_b})',
        f'volume ({asset_b})', f'amount ({asset_b})', 
        f'open ({asset_a})', f'close ({asset_a})', f'high ({asset_a})', f'low ({asset_a})', 
        f'volume ({asset_a})', f'amount ({asset_a})',
    ]
    scalers = [RobustScaler(), StandardScaler(), MinMaxScaler()]
    
    for scaler in scalers:
        scaler_name = scaler.__class__.__name__
        for col in cols_to_scale:
            if col in df.columns:
                df[col + "_" + scaler_name] = scaler.fit_transform(df[[col]])
            else:
                logging.warning(f"{col} not found in DataFrame")
    
    return df

def scale_and_annotate_samples(samples_with_metadata, symbol_a="ETH-USDT", symbol_b="BTC-USDT"):
    """
    Extracts data, timestamps, and labels from the input list of samples, applies the scaler function to the data,
    and returns a list of dictionaries with scaled data, timestamps, and labels.

    Args:
        samples_with_metadata (list of dict): List of samples, each containing 'data', 'timestamp', and 'label'.
        symbol_a (str): The symbol for the first asset (default is "ETH-USDT").
        symbol_b (str): The symbol for the second asset (default is "BTC-USDT").

    Returns:
        list of dict: List of dictionaries, each containing 'dataframe', 'timestamp', and 'label'.
    """

    data_list = [sample['data'] for sample in samples_with_metadata]
    timestamp_list = [sample['timestamp'] for sample in samples_with_metadata]
    label_list = [sample['label'] for sample in samples_with_metadata]

    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_index = {executor.submit(scale_dataframe, df, symbol_a, symbol_b): i for i, df in enumerate(data_list)}
        scaled_data = [None] * len(data_list)  # Placeholder for ordered results

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            scaled_data[idx] = future.result()

    scaled_samples_with_metadata = [
        {'dataframe': df, 'timestamp': ts, 'label': lbl} 
        for df, ts, lbl in zip(scaled_data, timestamp_list, label_list)
    ]
    
    return scaled_samples_with_metadata


def process_and_save_market_data(trade_type, profit_target, candle_span, collection_name, symbols=["ETH-USDT", "BTC-USDT"], timeframe="3min", main_symbol="ETH-USDT", head=None):
    """
    Processes market data for the given symbols and saves the processed samples to a specified collection.

    Parameters:
    symbols (list): List of market symbols to get data for.
    timeframe (str): Timeframe for the market data.
    main_symbol (str): The main symbol for which to label wins.
    trade_type (str): The type of trade ("long" or "short").
    profit_target (int): The profit target for labeling wins.
    collection_name (str): The name of the collection to save the processed samples to.
    """
    # logging.info("Starting market data processing...")

    # Step 1: Get market data
    if head:
        df = get_market_data_for_symbols(symbols, timeframe).head(head)
    else:
        df = get_market_data_for_symbols(symbols, timeframe)
    # logging.info("Market data retrieved successfully.")
    
    # Step 2: Drop unwanted columns early to save memory
    df = drop_unwanted_columns(df)
    # logging.info("Unwanted columns dropped.")
    
    # Force garbage collection to free up memory
    import gc
    gc.collect()

    # Step 3: Label wins
    df["win"] = label_wins(df, main_symbol, trade_type, candle_span, profit_target)
    # logging.info("Wins labeled.")

    # Force garbage collection to free up memory
    gc.collect()

    # Step 4: Extract previous candles with outcome
    samples_with_metadata = extract_previous_candles_with_outcome_in_batches(df)
    # logging.info("Previous candles with outcomes extracted.")

    # Release df from memory as it's no longer needed
    del df
    gc.collect()

    # Step 5: Scale and annotate samples
    scaled_samples_with_metadata = scale_and_annotate_samples(samples_with_metadata)
    # logging.info("Samples scaled and annotated.")

    # Step 6: Save samples to collection
    data_service = DataService(main_symbol, timeframe)
    data_service.save_samples_to_collection(samples_with_metadata=scaled_samples_with_metadata, collection_name=collection_name)
    # logging.info("Samples saved to collection successfully.")

    # Release remaining memory
    del scaled_samples_with_metadata
    del samples_with_metadata
    gc.collect()


def check_timestamps(timestamps):
    if not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(timestamps)
    timestamps = pd.to_datetime(timestamps)
    
    ascending = timestamps.is_monotonic_increasing
    time_diffs = timestamps.diff().dt.total_seconds()
    gaps = time_diffs[1:] > 180
    has_gaps = gaps.any()
    checks = {
        'ascending': ascending,
        'has_gaps': has_gaps,
        'gaps_indices': gaps[gaps].index.tolist()
    }
    if not checks['ascending'] or checks['has_gaps']:
        print(checks)
        raise ValueError("Timestamps are not in ascending order or there are gaps larger than 3 minutes.")
    return checks



def process_and_save_market_data_in_batches(trade_type, profit_target, candle_span, collection_name, symbols=["ETH-USDT", "BTC-USDT"], timeframe="3min", main_symbol="ETH-USDT", head=None, window_size=45, batch_size=1000):
    """
    Processes market data for the given symbols and saves the processed samples to a specified collection in batches.

    Parameters:
    symbols (list): List of market symbols to get data for.
    timeframe (str): Timeframe for the market data.
    main_symbol (str): The main symbol for which to label wins.
    trade_type (str): The type of trade ("long" or "short").
    profit_target (int): The profit target for labeling wins.
    collection_name (str): The name of the collection to save the processed samples to.
    batch_size (int): The size of each batch to process.
    """
    # logging.info("Starting market data processing...")

    # Step 1: Get market data
    if head:
        df = get_market_data_for_symbols(symbols, timeframe).head(head)
    else:
        df = get_market_data_for_symbols(symbols, timeframe)
    # logging.info("Market data retrieved successfully.")
    
    # Step 2: Drop unwanted columns early to save memory
    df = drop_unwanted_columns(df)
    # logging.info("Unwanted columns dropped.")
    
    # Force garbage collection to free up memory
    gc.collect()

    # Step 3: Label wins
    df["win"] = label_wins(df, main_symbol, trade_type, candle_span, profit_target)
    # logging.info("Wins labeled.")

    # Force garbage collection to free up memory
    gc.collect()
    print("outer")
    check_timestamps(df['timestamp'])
    # Process and save in batches
    num_batches = len(df) // batch_size + 1

    for batch_num in range(num_batches):
        start_idx = max(0, batch_num * batch_size - window_size)
        end_idx = min((batch_num + 1) * batch_size, len(df))

        # logging.info(f"Processing batch {batch_num + 1}/{num_batches}...")

        batch_df = df.iloc[start_idx:end_idx]
        # Step 4: Extract previous candles with outcome
        samples_with_metadata = extract_previous_candles_with_outcome(batch_df, window_size)
        # logging.info("Previous candles with outcomes extracted.")
        print("inner_1")
        timestamps = [sample['timestamp'] for sample in samples_with_metadata]
        check_timestamps(timestamps)
        # Release batch_df from memory as it's no longer needed
        del batch_df
        gc.collect()

        # Step 5: Scale and annotate samples
        scaled_samples_with_metadata = scale_and_annotate_samples(samples_with_metadata)
        # logging.info("Samples scaled and annotated.")
        print("inner_2")
        timestamps = [sample['timestamp'] for sample in scaled_samples_with_metadata]
        check_timestamps(timestamps)
        # Step 6: Save samples to collection
        data_service = DataService(main_symbol, timeframe)
        print("inner_3")
        timestamps = [sample['timestamp'] for sample in scaled_samples_with_metadata]
        check_timestamps(timestamps)
        data_service.save_samples_to_collection(samples_with_metadata=scaled_samples_with_metadata, collection_name=collection_name)
        # logging.info("Samples saved to collection successfully.")

        # Release memory for the batch
        del scaled_samples_with_metadata
        del samples_with_metadata
        gc.collect()

    # logging.info("All batches processed and saved successfully.")

def scale_and_cleanup_dataframe(df: pd.DataFrame, selected_scaler: str = 'standard') -> pd.DataFrame:
    """
    Scales and cleans up the DataFrame by replacing unscaled columns with their scaled counterparts
    and then dropping the scaled columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to be scaled and cleaned up.
    selected_scaler (str): The scaler type to use. Must be one of 'minmax', 'standard', or 'robust'.
                           Default is 'standard'.

    Returns:
    pd.DataFrame: The cleaned-up DataFrame with the unscaled columns replaced by their scaled counterparts
                  and the scaled columns dropped.

    Raises:
    ValueError: If the selected_scaler is not in the list of available scalers.
    """
    available_scalers = ['minmax', 'standard', 'robust']
    
    # Ensure the selected scaler is in the available scalers list
    if selected_scaler.lower() not in available_scalers:
        raise ValueError(f"Selected scaler '{selected_scaler}' is not in the list of available scalers: {available_scalers}")

    # Create a list of columns that are not scaled
    unscaled_cols = [col for col in df.columns if not any(scaler in col.lower() for scaler in available_scalers)]

    # Create a list of columns that are scaled
    scaled_cols = [col for col in df.columns if any(scaler in col.lower() for scaler in available_scalers)]
    
    # Replace unscaled columns with their scaled counterparts
    for col in unscaled_cols:
        scaled_col_name = col + "_" + selected_scaler
        if scaled_col_name in scaled_cols:
            df[col] = df[scaled_col_name]
    
    # Drop scaled columns
    df = df.drop(scaled_cols, axis=1)
    
    return df
