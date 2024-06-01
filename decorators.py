import time
import csv
import os
from datetime import datetime
from functools import wraps
from format import format_time

def time_it(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        # Ensure the 'logs' directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Define the path to the log file
        log_file_path = os.path.join('logs', 'optimization_log.csv')
        
        # Record the start time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        # Write log entry to the file
        with open(log_file_path, "a", newline="") as f:
            msg = (func.__name__, end - start, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), args, kwargs)
            writer = csv.writer(f)
            writer.writerow(msg)
            print(msg)
        
        return result
    
    return wrap