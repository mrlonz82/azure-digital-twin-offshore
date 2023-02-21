import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data from CSV file
data = pd.read_csv('power/Q1.csv')

# Drop any missing values
data.dropna(inplace=True)

# Define the feature and target columns
X = data[['wind_speed']]
y = data[['kW']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Decision Tree Regression
dt = DecisionTreeRegressor(random_state=0)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Support Vector Regression
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train.values.ravel())
svr_pred = svr.predict(X_test)

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train, y_train.values.ravel())
rf_pred = rf.predict(X_test)

# KN KNeighborsRegressor
kn = KNeighborsRegressor()
kn.fit(X_train, y_train.values.ravel())
kn_pred = kn.predict(X_test)

# Evaluate the models using MSE, MAE, and RMSE
print('Decision Tree Regression:')
print('MSE:', mean_squared_error(y_test, dt_pred))
print('MAE:', mean_absolute_error(y_test, dt_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, dt_pred)))
print('R2 Score:', r2_score(y_test, dt_pred))
print('')

print('Support Vector Regression:')
print('MSE:', mean_squared_error(y_test, svr_pred))
print('MAE:', mean_absolute_error(y_test, svr_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, svr_pred)))
print('R2 Score:', r2_score(y_test, svr_pred))
print('')

print('Random Forest Regression:')
print('MSE:', mean_squared_error(y_test, rf_pred))
print('MAE:', mean_absolute_error(y_test, rf_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, rf_pred)))
print('R2 Score:', r2_score(y_test, rf_pred))
print('')

print('kNN regression:')
print('MSE:', mean_squared_error(y_test, kn_pred))
print('MAE:', mean_absolute_error(y_test, kn_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, kn_pred)))
print('R2 Score:', r2_score(y_test, kn_pred))
print('')


# Plot the predicted vs. actual values
plt.plot(y_train.values.ravel(), label='Actual')
plt.plot(dt_pred, label='Prediction')
plt.title('Power Generation Prediction (Decision Tree Model)')
plt.xlabel('Samples')
plt.ylabel('Power')
plt.legend()
plt.show()

plt.plot(y_train.values.ravel(), label='Actual')
plt.plot(svr_pred, label='Prediction')
plt.title('Power Generation Prediction (Support Vector)')
plt.xlabel('Samples')
plt.ylabel('Power')
plt.legend()
plt.show()


plt.plot(y_train.values.ravel(), label='Actual')
plt.plot(rf_pred, label='Prediction')
plt.title('Power Generation Prediction (Random Forest)')
plt.xlabel('Samples')
plt.ylabel('Power')
plt.legend()
plt.show()

plt.plot(y_train.values.ravel(), label='Actual')
plt.plot(rf_pred, label='Prediction')
plt.title('Power Generation Prediction (kNN regression)')
plt.xlabel('Samples')
plt.ylabel('Power')
plt.legend()
plt.show()

