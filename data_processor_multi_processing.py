from data_service import DataService
from labeler import BinaryWinFinder 
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

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
        print(df, win_value, timestamp_value)
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
    logging.info("Starting market data processing...")

    # Step 1: Get market data
    if head:
        df = get_market_data_for_symbols(symbols, timeframe).head(head)
    else:
        df = get_market_data_for_symbols(symbols, timeframe)
    logging.info("Market data retrieved successfully.")

    # Step 2: Drop unwanted columns
    df = drop_unwanted_columns(df)
    print(df)
    logging.info("Unwanted columns dropped.")

    # Step 3: Label wins
    df["win"] = label_wins(df, main_symbol, trade_type, candle_span, profit_target)
    logging.info("Wins labeled.")

    # Step 4: Extract previous candles with outcome
    samples_with_metadata = extract_previous_candles_with_outcome(df)
    logging.info("Previous candles with outcomes extracted.")

    # Step 5: Scale and annotate samples
    scaled_samples_with_metadata = scale_and_annotate_samples(samples_with_metadata)
    logging.info("Samples scaled and annotated.")

    # Step 6: Save samples to collection
    data_service = DataService(main_symbol, timeframe)
    data_service.save_samples_to_collection(samples_with_metadata=scaled_samples_with_metadata, collection_name=collection_name)
    logging.info("Samples saved to collection successfully.")


# Ensure to define your DataService, BinaryWinFinder, and other required classes and methods.
