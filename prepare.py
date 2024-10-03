import pandas as pd

df = pd.read_csv('data/bikeshare.txt')

# drop missing values
initial_count = len(df)
df.dropna(inplace=True)
final_count = len(df)
print(f"Number of rows dropped: {initial_count - final_count}")


# convert the 'dteday' and 'hr' columns to a datetime object
df['datetime'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
df.set_index('datetime', inplace=True)

# drop the unnecessary columns
df = df.drop(columns=['instant', 'hr', 'yr','mnth','dteday'])

df.to_csv('data/bikeshare_prepared.txt')