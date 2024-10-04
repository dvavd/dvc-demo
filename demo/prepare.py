import pandas as pd
import os


print("Current working directory:", os.getcwd())

training_data = './data/bikesharing/train/bikeshare_v2.0.txt'
validation_data = './data/bikesharing/validation/validation.txt'

training_prepared = './data/bikesharing/train/bikeshare_prepared.txt'
validation_prepared = './data/bikesharing/validation/validation_prepared.txt'

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
    df = df.drop(columns=['instant', 'yr', 'mnth', 'hr', 'season', 'atemp', 'hum', 'windspeed'])

    # Export data to path
    df.to_csv(output_path)

data_cleaning(training_data, training_prepared)
data_cleaning(validation_data, validation_prepared)

#plt.savefig('./plots/scatter_plot.png')
