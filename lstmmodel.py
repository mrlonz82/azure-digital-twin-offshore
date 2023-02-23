# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


def _create_dataset(X, y, look_back):
    # create sliding window dataset
    X_data, y_data = [], []
    for i in range(len(X) - look_back):
        X_data.append(X[i:(i + look_back)])
        y_data.append(y[i + look_back])
    return np.array(X_data), np.array(y_data)


def _split_data_to_training_and_testing(data: pd.DataFrame, Y_col, X_col):
    look_back = 12
    _X, y = data[X_col].values, data[Y_col].values
    _X, y = _create_dataset(_X, y, look_back)
    train_size = int(len(_X) * 0.7)
    X_train, X_test = _X[:train_size], _X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test


def _check_model_performance(title: str, y_test, y_pred):
    # calculate performance.txt metrics
    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)

    # Open a file in append mode
    with open("performance.txt", "a") as f:
        # Write some text to the file
        f.write(f"---------------LSTM Model Performance for {title}---------------\n")
        f.write(f'{title} Test MSE: %.2f\n' % test_mse)
        f.write(f'{title} Test MAE: %.2f\n' % test_mae)
        f.write(f'{title} Test RMSE: %.2f\n' % test_rmse)
        f.write("-----------------------------------------------------------------\n")

    print(f'{title} Test MSE: %.2f' % test_mse)
    print(f'{title} Test MAE: %.2f' % test_mae)
    print(f'{title} Test RMSE: %.2f' % test_rmse)


def _plot_result(title: str, y_test, y_pred, path_to_save: str):
    # plot wind speed predictions
    plt.plot(y_test, label=f'Actual {title}')
    plt.plot(y_pred, label=f'Predicted {title}')
    plt.legend()
    plt.title(f"{title}")
    plt.savefig(f"plots/{path_to_save}/{title}.png")
    plt.close()


def train_and_predict(title: str, path_to_data: str, y_col: str, x_col: str, path_to_save: str):
    # load data
    data = pd.read_csv(path_to_data)

    # preprocess data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[f"scaled{y_col}"] = scaler.fit_transform(data[y_col].values.reshape(-1, 1))
    X_train, X_test, y_train, y_test = _split_data_to_training_and_testing(data, y_col, x_col)

    # create LSTM model for wind speed prediction
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # train
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # make predictions on test data
    y_pred = model.predict(X_test)

    # inverse transform scaled data to original scale
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

    _check_model_performance(title, y_test, y_pred)
    _plot_result(title, y_test, y_pred, path_to_save)

    print("----The End----")


# wind_speed
train_and_predict("Wind Speed Q1", "datasets/splits/Q1.csv", "wind speed", "wind speed", "lstm")
train_and_predict("Wind Speed Q2", "datasets/splits/Q2.csv", "wind speed", "wind speed", "lstm")
train_and_predict("Wind Speed Q3", "datasets/splits/Q3.csv", "wind speed", "wind speed", "lstm")
train_and_predict("Wind Speed Q4", "datasets/splits/Q4.csv", "wind speed", "wind speed", "lstm")

# wind_direction
train_and_predict("Wind direction Q1", "datasets/splits/Q1.csv", "wind direction", "wind direction", "lstm")
train_and_predict("Wind direction Q2", "datasets/splits/Q2.csv", "wind direction", "wind direction", "lstm")
train_and_predict("Wind direction Q3", "datasets/splits/Q3.csv", "wind direction", "wind direction", "lstm")
train_and_predict("Wind direction Q4", "datasets/splits/Q4.csv", "wind direction", "wind direction", "lstm")
