from datetime import datetime

# Define a dictionary for various time formats
time_formats = {
    "timestamp": "%Y-%m-%d_%H-%M-%S",
    "date": "%Y-%m-%d",
    "time": "%H:%M:%S",
    "verbose": "%A, %d %B %Y %I:%M%p"
}

# Function to get formatted time
def format_time(time_input, format_key):
    if format_key not in time_formats:
        raise ValueError(f"Format '{format_key}' not found in time_formats dictionary.")
    if isinstance(time_input, datetime):
        time_obj = time_input
    else:
        time_obj = datetime.fromtimestamp(time_input)
    return time_obj.strftime(time_formats[format_key])