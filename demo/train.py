import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

bike = pd.read_csv('./data/bikesharing/train/bikeshare_prepared.txt')
test = pd.read_csv('./data/bikesharing/validation/validation_prepared.txt')

# Calculate the 'prop_casual'
bike['prop_casual'] = bike['casual'] / bike['cnt']
test['prop_casual'] = test['casual'] / test['cnt']

# Prepare the features and target variable of training & testing
X_train = bike[['temp']]  # Features (independent variable)
y_train = bike['prop_casual']

X_test = test[['temp']]
y_test = test['prop_casual']

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Validation Results')
print(f'Mean Squared Error: {mse}')
print(f'R-squared Value: {r2}')

plt.figure(figsize=(10, 6))
sns.lineplot(x=X_test['temp'], y=y_test, label='Actual', color='blue')
sns.lineplot(x=X_test['temp'], y=y_pred, label='Predicted', color='red')
plt.title('Linear Regression: Temperature vs Casual Rider Proportion')
plt.ylabel('Proportion of Casual Riders')
plt.xlabel('Temperature')
plt.legend()
plt.savefig('./plots/model_plot.png')
