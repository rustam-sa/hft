# candlestick plot functions
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import seaborn as sns


def create_candlestick_grid(data_list, rows, cols, filename='candlestick_grid.png'):
    sns.set(style="whitegrid")

    # Calculate figure size for office paper (8.5 x 11 inches) at 300 DPI
    fig_width = 8.5
    fig_height = 11
    dpi = 300

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=dpi)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    for ax, data in zip(axes, data_list):
        # Convert timestamp to datetime
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
        
        # Prepare data for candlestick plotting
        btc_data = data[['timestamp', 'open (BTC)', 'high (BTC)', 'low (BTC)', 'close (BTC)']].copy()
        eth_data = data[['timestamp', 'open (ETH)', 'high (ETH)', 'low (ETH)', 'close (ETH)']].copy()
        
        # Convert timestamp to Matplotlib date format
        btc_data['timestamp'] = btc_data['timestamp'].apply(mdates.date2num)
        eth_data['timestamp'] = eth_data['timestamp'].apply(mdates.date2num)
        
        # Plot BTC candlestick chart
        candlestick_ohlc(ax, btc_data.values, width=0.01, colorup='green', colordown='red')
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_title('BTC-USD')
        ax.set_ylabel('Price (BTC)')
        ax.set_xlabel('Time')
        plt.xticks(rotation=45)
    
    # Hide any remaining empty subplots
    for i in range(len(data_list), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    

def create_dual_candlestick_plot(data, symbol1, symbol2, y_range1=None, y_range2=None):
    # Convert timestamp to datetime
    data = data.copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    
    # Prepare data for candlestick plotting
    symbol1_data = data[['timestamp', f'open ({symbol1})', f'high ({symbol1})', f'low ({symbol1})', f'close ({symbol1})']].copy()
    symbol2_data = data[['timestamp', f'open ({symbol2})', f'high ({symbol2})', f'low ({symbol2})', f'close ({symbol2})']].copy()
    
    # Convert timestamp to Matplotlib date format
    symbol1_data['timestamp'] = symbol1_data['timestamp'].apply(mdates.date2num)
    symbol2_data['timestamp'] = symbol2_data['timestamp'].apply(mdates.date2num)
    
    sns.set_theme(style="darkgrid")
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)
    
    # Plot symbol1 candlestick chart
    candlestick_ohlc(ax1, symbol1_data.values, width=0.01, colorup='green', colordown='red')
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax1.set_title(f'{symbol1} Candlestick Chart')
    ax1.set_ylabel(f'Price ({symbol1})')
    
    # Set y-axis limits for symbol1 chart if provided
    if y_range1:
        ax1.set_ylim(y_range1)
    
    # Plot symbol2 candlestick chart
    candlestick_ohlc(ax2, symbol2_data.values, width=0.01, colorup='blue', colordown='orange')
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax2.set_title(f'{symbol2} Candlestick Chart')
    ax2.set_ylabel(f'Price ({symbol2})')
    ax2.set_xlabel('Timestamp')
    
    # Set y-axis limits for symbol2 chart if provided
    if y_range2:
        ax2.set_ylim(y_range2)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def create_minimal_candlestick_plot(data, symbol1, symbol2, y_range1=None, y_range2=None, image_size=(64, 64)):
    # Convert timestamp to datetime
    # Convert timestamp to datetime
    symbol1 = symbol1.split("-")[0].lower()
    symbol2 = symbol2.split("-")[0].lower()
    data = data.copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    
    # Prepare data for candlestick plotting
    symbol1_data = data[['timestamp', f'open_{symbol1}', f'high_{symbol1}', f'low_{symbol1}', f'close_{symbol1}']].copy()
    symbol2_data = data[['timestamp', f'open_{symbol2}', f'high_{symbol2}', f'low_{symbol2}', f'close_{symbol2}']].copy()
    
    # Convert timestamp to Matplotlib date format
    symbol1_data['timestamp'] = symbol1_data['timestamp'].apply(mdates.date2num)
    symbol2_data['timestamp'] = symbol2_data['timestamp'].apply(mdates.date2num)
    
    sns.set_theme(style="darkgrid")
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=image_size, dpi=100)
    ax1, ax2 = axes
    
    # Plot symbol1 candlestick chart
    candlestick_ohlc(ax1, symbol1_data.values, width=0.004, colorup='green', colordown='red')
    ax1.axis('off')  # Turn off axis labels, ticks, and grid
    if y_range1:
        ax1.set_ylim(y_range1)
    
    # Plot symbol2 candlestick chart
    candlestick_ohlc(ax2, symbol2_data.values, width=0.004, colorup='blue', colordown='orange')
    ax2.axis('off')  # Turn off axis labels, ticks, and grid
    if y_range2:
        ax2.set_ylim(y_range2)

    
    plt.subplots_adjust(wspace=0, hspace=0)  # Remove space between subplots
    plt.tight_layout(pad=0)  # Remove padding
    plt.close(fig)
    
    return fig


def save_plot(plot, output_dir, sample_name):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Save the figure as an image file
    plot_filename = os.path.join(output_dir, f'candlestick_plot_{sample_name}.png')
    plot.savefig(plot_filename, bbox_inches='tight', pad_inches=0)
    plt.clf()  # Clear the plot from memory
    plt.close(plot)
    return plot_filename
