def create_segmentation_map(cross_val_count, train_segment_size, val_segment_size, test_segment_size):
    """
    Creates a segmentation table for organizing data into different sets for cross-validation.
    This function assumes data has a temporally ascending index.
    
    Parameters:
    - cross_val_count (int): The number of cross-validation folds.
    - train_segment_size (int): The size of each training segment.
    - scaler_size (int): The size of the segment used for scaling features.
    - val_segment_size (int): The size of each validation segment.
    - test_segment_size (int): The size of each test segment.
    
    Returns:
    - dict: A dictionary where keys are segment names and values are tuples indicating the
            start and end indices of each segment.
    """


    segmentation_map = {}
    segmentation_map['total_data_needed'] = train_segment_size + val_segment_size + test_segment_size * 3
    for n in range(cross_val_count):
        moving_window_size = n * test_segment_size
        train_segment_start = 0 + moving_window_size
        train_segment_end = train_segment_size + moving_window_size
        segmentation_map[f'train_segment_{n}'] = train_segment_start, train_segment_end
        

        
        val_segment_start = train_segment_end
        val_segment_end = train_segment_end + val_segment_size
        segmentation_map[f'val_segment_{n}'] = val_segment_start, val_segment_end
        
        test_segment_start = val_segment_end
        test_segment_end = val_segment_end + test_segment_size
        segmentation_map[f'test_segment_{n}'] = test_segment_start, test_segment_end
        
        evaluation_segment_start = val_segment_start
        evaluation_segment_end = test_segment_end
        segmentation_map[f'evaluation_segment_{n}'] = evaluation_segment_start, evaluation_segment_end
        
    
    return segmentation_map


def get_segment(segment, start, end):
    """
    Extracts a sub-segment from the given segment based on start and end indices.
    
    Parameters:
    - segment (pd.DataFrame): The input segment from which to extract the sub-segment.
    - start (int): The start index for the sub-segment extraction.
    - end (int): The end index for the sub-segment extraction.
    
    Returns:
    - pd.DataFrame: The extracted sub-segment, reversed and reset.
    """


    sub_segment = segment.copy().reset_index(drop=True)
    sub_segment = sub_segment.iloc[start:end].reset_index(drop=True)
    return sub_segment