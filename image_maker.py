import os
import pandas as pd
from pathlib import Path
import data_processor_multi_processing as processor
import plotter
from data_service import DataService

# Function to get range of values
def get_ranges(symbol, list_of_dataframes):
    symbol = symbol.split("-")[0].lower()
    range_max = float('-inf')
    range_min = float('inf')
    
    for df in list_of_dataframes:
        high_col = f"high_{symbol}"
        low_col = f"low_{symbol}"
        
        if high_col in df and low_col in df:
            range_max = max(range_max, df[high_col].max())
            range_min = min(range_min, df[low_col].min())

    return range_min, range_max

# Function to replace columns with scaled columns
def replace_cols_with_scaled_cols(samples, selected_scaler='standard'):
    for s in samples:
        new_dataframe = processor.scale_and_cleanup_dataframe(s['data'], selected_scaler)
        s.update({'data': new_dataframe})
    return samples

# Function to check timestamps for order and gaps
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

# Function to split list into segments
def split_list(data, fractions):
    if len(fractions) != 3:
        raise ValueError("Fractions list must contain exactly 3 elements.")
    if sum(fractions) != 1:
        raise ValueError("Fractions must sum to 1.")
    
    total_length = len(data)
    lengths = [int(total_length * fraction) for fraction in fractions]
    while sum(lengths) < total_length:
        lengths[lengths.index(min(lengths))] += 1
    
    segment1 = data[:lengths[0]]
    segment2 = data[lengths[0]:lengths[0] + lengths[1]]
    segment3 = data[lengths[0] + lengths[1]:]

    return segment1, segment2, segment3

# Function to check temporal order
def check_temporal_order(train, val, test):
    train_timestamps = pd.to_datetime([d['timestamp'] for d in train])
    val_timestamps = pd.to_datetime([d['timestamp'] for d in val])
    test_timestamps = pd.to_datetime([d['timestamp'] for d in test])
    
    if not (train_timestamps.is_monotonic_increasing and val_timestamps.is_monotonic_increasing and test_timestamps.is_monotonic_increasing):
        return False
    
    if train_timestamps.max() >= val_timestamps.min() or val_timestamps.max() >= test_timestamps.min():
        return False
    
    return True


def process_and_plot_data(collection_name, fractions, output_dir):
    """
    Processes data from a specific collection and generates candlestick plots.

    Parameters:
    collection_index (int): Index of the collection to process.
    fractions (list): List of 3 fractions representing the relative sizes of the train, validation, and test segments.
    output_directory (str): Directory to save the generated plots.
    """
    ds = DataService()
    data = ds.get_dataframes_as_dicts(collection_name=collection_name)
    
    # Check timestamps
    timestamps = pd.Series([s['timestamp'] for s in data])
    checks = check_timestamps(timestamps)
    if not checks['ascending'] or checks['has_gaps']:
        raise ValueError("Timestamps are not in ascending order or there are gaps larger than 3 minutes.")
    
    # Split data
    train, val, test = split_list(data, fractions=fractions)
    check = check_temporal_order(train, val, test)
    if not check:
        raise ValueError("Train, validation, and test sets are not properly temporally separated.")
    for samples in train, val, test:
        samples = replace_cols_with_scaled_cols(samples)
    # Get ranges
    dataframes = [s['data'] for s in train]
    range_min_eth, range_max_eth = get_ranges("ETH-USDT", dataframes)
    range_min_btc, range_max_btc = get_ranges("BTC-USDT", dataframes)
    
    # Plotting
    os.makedirs(output_dir, exist_ok=False)
    sub_dir_names = ['train', 'val', 'test']
    
    for i, dataset in enumerate([train, val, test]):
        labels = []
        new_output_dir = Path(output_dir) / f"{sub_dir_names[i]}"
        for sample in dataset:
            plot = plotter.create_minimal_candlestick_plot(
                sample['data'], 'ETH-USDT', 'BTC-USDT',
                y_range1=(range_min_eth, range_max_eth),
                y_range2=(range_min_btc, range_max_btc)
            )
            labels.append(sample['label'])
            plotter.save_plot(plot, new_output_dir, sample['timestamp'])
        labels = pd.Series(labels)
        labels.to_csv(new_output_dir +"/labels.csv")

