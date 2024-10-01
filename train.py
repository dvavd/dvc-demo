import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('data/bikeshare_prepared.txt', index_col=0)

X = df.drop(columns=['cnt'])
y = df['cnt']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'model.pkl')
