import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from statsmodels.nonparametric.smoothers_lowess import lowess


# Default plot configurations
plt.rcParams['figure.figsize'] = (16,8)
plt.rcParams['figure.dpi'] = 150
plt.figure(figsize=(8,6))
sns.set()

bike = pd.read_csv('prepared_data/bikeshare_prepared.txt')

# Modify holiday, weekday, workingday, and weather state here
factor_dict = {
    'holiday': {0: "no", 1: "yes"},
    'weekday': {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed",
                4: "Thu", 5: "Fri", 6: "Sat"},
    'workingday': {0: "no", 1: "yes"},
    'weather': {1: "Clear", 2: "Mist",
                3: "Light", 4: "Heavy"},
}
bike.replace(factor_dict, inplace=True)
bike.head()

# Group the data by dates
daily_counts = bike.groupby('date').agg({'casual': 'sum', 'registered': 'sum', 'workingday': 'first'})

# The ratio of casual riders in total cnt
bike['prop_casual'] = bike['casual']/bike['cnt']

# Classify the days into 'weekday' or 'weekend'
bike['day_type'] = bike['weekday'].apply(
    lambda x: 'weekdays' if x in ["Mon", "Tue", "Wed", "Thu", "Fri"] else 'weekends'
)

# Filter out potential outliers
bike_filtered = bike[(bike['prop_casual'] > 0) & (bike['prop_casual'] < 1)]

# Temperature Comparison Smoothed Line Plot ========================
for day_type in ['weekdays', 'weekends']:
    # Filter data based on 'day_type'
    day_type_prop = bike_filtered[bike_filtered['day_type'] == day_type][['prop_casual', 'temp']]
    prop = day_type_prop

    # Apply LOWESS smoothing
    ysmooth = lowess(day_type_prop['prop_casual'], day_type_prop['temp'], return_sorted=False)

    # Plot the smoothed line
    sns.lineplot(day_type_prop['temp'].to_numpy(), ysmooth, label=day_type)
# ==================================================================

# Temperature Comparison Scatter Plot ==============================
for day_type in ['weekdays', 'weekends']:
    # Sample n data from dataset for plotting
    sampled_data = bike_filtered[bike_filtered['day_type'] == day_type].sample(n = 200)

    xobs = (sampled_data ['temp']).to_numpy()
    yobs = sampled_data ['prop_casual'].to_numpy()

    # Plot with sampled data points
    sns.scatterplot(xobs, yobs, label="Raw Data")
# ==================================================================


plt.title("Temperature vs. Casual Rider Proportion by Workdays")
plt.ylabel("Casual Rider Proportion")
plt.xlabel("Temperature(Celsius)")
plt.legend()

"""
# Distribution Comparison Histogram
sns.histplot(data=daily_counts[["casual"]], stat='density', kde=True, label="casual")
sns.histplot(data=daily_counts[["registered"]], stat='density', kde=True, 
             palette=sns.xkcd_palette(["green"]), label="registered")

plt.title("Distribution Comparison of Casual vs. Registered Riders")
plt.xlabel("Rider Count")
plt.legend()
"""

plt.savefig('plot.png')
plt.show()
