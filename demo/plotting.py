import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from statsmodels.nonparametric.smoothers_lowess import lowess

def _set_plot():
    # Default plot configurations
    plt.rcParams['figure.figsize'] = (16, 8)
    plt.rcParams['figure.dpi'] = 150
    plt.figure(figsize=(10, 6))
    sns.set()

def line_plot(data, show=True, save=False, reset=True):
    if reset: _set_plot()
    # Temperature Comparison Smoothed Line Plot ====================
    for day_type in ['weekdays', 'weekends']:
        # Filter data based on 'day_type'
        day_type_prop = data[data['day_type'] == day_type][['prop_casual', 'temp']]

        # Apply LOWESS smoothing
        ysmooth = lowess(day_type_prop['prop_casual'], day_type_prop['temp'], return_sorted=False)

        # Plot the smoothed line
        sns.lineplot(x=day_type_prop['temp'].to_numpy(), y=ysmooth, label=day_type)
    # ==============================================================
    if show or save:
        plt.title("Temperature vs. Casual Rider Proportion by Workdays")
        plt.ylabel("Casual Rider Proportion")
        plt.xlabel("Temperature(Celsius)")
        plt.legend()
        if save: plt.savefig('linear_plot.png')
        if show: plt.show()
        plt.close()

def scatter_plot(data, show=True, save=False, reset=True):
    if reset: _set_plot()
    # Temperature Comparison Scatter Plot ==========================
    for day_type in ['weekdays', 'weekends']:
        # Sample n data from dataset for plotting
        sampled_data = data[data['day_type'] == day_type].sample(n = 200)

        xobs = (sampled_data ['temp']).to_numpy()
        yobs = sampled_data ['prop_casual'].to_numpy()

        # Plot with sampled data points
        sns.scatterplot(x=xobs, y=yobs, label="Raw Data")
    # ==============================================================
    if show or save:
        plt.title("Temperature vs. Casual Rider Proportion by Workdays")
        plt.ylabel("Casual Rider Proportion")
        plt.xlabel("Temperature(Celsius)")
        plt.legend()
        if save: plt.savefig('scatter_plot.png')
        if show: plt.show()
        plt.close()

def combined_plot(data, show=True, save=False, reset=True):
    if reset: _set_plot()
    # Temperature Comparison Combined Plot =========================
    line_plot(data, show=False, save=False)
    scatter_plot(data, show=False, save=False, reset=False)
    # ==============================================================
    if show or save:
        plt.title("Temperature vs. Casual Rider Proportion by Workdays")
        plt.ylabel("Casual Rider Proportion")
        plt.xlabel("Temperature(Celsius)")
        plt.legend()
        if save: plt.savefig('combined_plot.png')
        if show: plt.show()
        plt.close()

def histogram_plot(data, show=True, save=False, reset=True):
    if reset: _set_plot()
    # Group the data by dates
    daily_counts = data.groupby('date').agg({'casual': 'sum', 'registered': 'sum', 'workingday': 'first'})
    # Distribution Comparison Histogram ============================
    sns.histplot(data=daily_counts[["casual"]], stat='density', kde=True, label="casual")
    sns.histplot(data=daily_counts[["registered"]], stat='density', kde=True,
                 palette=sns.xkcd_palette(["green"]), label="registered")
    # ==============================================================
    if show or save:
        plt.title("Distribution Comparison of Casual vs. Registered Riders")
        plt.xlabel("Rider Count")
        plt.legend()
        if save: plt.savefig('histogram_plot.png')
        if show: plt.show()
        plt.close()
