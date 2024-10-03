import numpy as np
import pandas as pd

bike = pd.read_csv('train/bikeshare_v2.0.txt')

# Calculate the ratio of casual/(casual+registered)
bike['casual_ratio'] = bike['casual'] / bike['cnt']

# Define a mask for filtering the rows
mask = bike['casual_ratio'] > 0.3

# Randomly drop rows with 80% chance
drop_mask = np.random.choice([True, False], size=mask.sum(), p=[0.8, 0.2])

# Create a filter mask and apply it to filter the DataFrame
mask[mask] = drop_mask
bike_filtered = bike[~mask]

# Drop the temp 'casual_ratio' column
bike_filtered = bike_filtered.drop(columns=['casual_ratio'])

# Reorder 'instant' column by resetting the index
bike_filtered = bike_filtered.reset_index(drop=True)
bike_filtered['instant'] = bike_filtered.index + 1

# Export the filtered data to a .txt file
bike_filtered.to_csv('bikeshare_v1.0.txt', sep=',', index=False)
