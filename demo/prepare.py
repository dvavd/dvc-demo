import pandas as pd
import plotting
import os


print("Current working directory:", os.getcwd())

training_data = '../data/bikesharing/train/bikeshare_v2.0.txt'
validation_data = '../data/bikesharing/validation/validation.txt'

training_prepared = '../data/bikesharing/train/bikeshare_prepared.txt'
validation_prepared = '../data/bikesharing/validation/validation_prepared.txt'

def data_cleaning(data_path, output_path):
    # Create a dataframe for cleaning
    df = pd.read_csv(data_path)

    # Convert the 'dteday' and 'hr' columns to a datetime object
    df['datetime'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    df.set_index('datetime', inplace=True)

    # Recover temperature data
    df['temp'] = (df['temp'] * 41).round(2)

    # Rename the columns
    df = df.rename(columns={
        'dteday': 'date',
        'weathersit': 'weather'
    })
    # Drop the unnecessary columns
    columns_to_keep = ['date', 'workingday', 'temp', 'weather', 'casual', 'registered', 'cnt']
    df = df[columns_to_keep]

    # Export data to path
    df.to_csv(output_path)

    # Modify weather state here
    factor_dict = {
        'weather': {1: "Clear", 2: "Mist", 3: "Light", 4: "Heavy"},
    }
    df.replace(factor_dict, inplace=True)
    df.head()

    # Classify the days into 'weekday' or 'weekend'
    df['day_type'] = df['workingday'].apply(
        lambda x: 'weekdays' if x == 0 else 'weekends'
    )

    # The ratio of casual riders in total cnt
    df['prop_casual'] = df['casual'] / df['cnt']
    # Filter out potential outliers
    df = df[(df['prop_casual'] > 0) & (df['prop_casual'] < 1)]

    return df


if __name__ == '__main__':
    # Step 1: Clean the data
    data = data_cleaning(training_data, training_prepared)
    data_cleaning(validation_data, validation_prepared)

    # Step 2: Overview of data
    plotting.scatter_plot(data, show=False, save=True)
    plotting.combined_plot(data, show=True, save=False)
