import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from tcn import TCN
from keras.layers import LSTM, Dense, Flatten


def _create_model(model: str, X_train, y_train, window_size) -> Sequential | TCN:
    # create LSTM model for  prediction
    if model == "lstm":
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
    else:

        # create TCN model for  prediction
        model = Sequential([
            TCN(input_shape=(window_size, 1), nb_filters=64, kernel_size=3, dropout_rate=0.2),
            Flatten(),
            Dense(1)
        ])
    return model


def _create_dataset(X, y, look_back):
    # create sliding window dataset
    X_data, y_data = [], []
    for i in range(len(X) - look_back):
        X_data.append(X[i:(i + look_back)])
        y_data.append(y[i + look_back])
    return np.array(X_data), np.array(y_data)


def _split_data_to_training_and_testing(model: str, data: pd.DataFrame, Y_col, X_col, window_size: int,
                                        scaler: MinMaxScaler):
    if model == "lstm":
        scaled_y_col = f"scaled_{Y_col}"
        data[scaled_y_col] = scaler.fit_transform(data[Y_col].values.reshape(-1, 1))
        look_back = 12
        _X, y = data[X_col].values, data[Y_col].values
        _X, y = _create_dataset(_X, y, look_back)
        train_size = int(len(_X) * 0.7)
        X_train, X_test = _X[:train_size], _X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
    else:
        data = scaler.fit_transform(data[Y_col].values.reshape(-1, 1))
        train_size = int(len(data) * 0.8)
        train_data, test_data = data[:train_size], data[train_size:]
        # Create the input and target data for the TCN model
        X_train, y_train = [], []
        for i in range(window_size, len(train_data)):
            X_train.append(train_data[i - window_size:i, 0])
            y_train.append(train_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        X_test, y_test = [], []
        for i in range(window_size, len(test_data)):
            X_test.append(test_data[i - window_size:i, 0])
            y_test.append(test_data[i, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)
    return X_train, X_test, y_train, y_test


def _check_model_performance(model, title: str, y_test, y_pred):
    # calculate performance.txt metrics
    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)

    # Open a file in append mode
    with open("performance.txt", "a") as f:
        # Write some text to the file
        f.write(f"---------------{model} Model Performance for {title}---------------\n")
        f.write(f'{title} Test MSE: %.2f\n' % test_mse)
        f.write(f'{title} Test MAE: %.2f\n' % test_mae)
        f.write(f'{title} Test RMSE: %.2f\n' % test_rmse)
        f.write("-----------------------------------------------------------------\n")

    print(f'{title} Test MSE: %.2f' % test_mse)
    print(f'{title} Test MAE: %.2f' % test_mae)
    print(f'{title} Test RMSE: %.2f' % test_rmse)


def _plot_result(title: str, y_test, y_pred, path_to_save: str, test_data: pd.DataFrame, label: str):
    # plot
    fig = plt.figure(figsize=(10, 5))
    dates = test_data.index
    plt.plot(dates, y_test, label=f'Actual {title}', scalex=False)
    plt.plot(dates, y_pred, label=f'Predicted {title}')
    date_format = '%d'
    formatter = dt.DateFormatter(date_format)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xticks()
    plt.locator_params(axis="x", nbins=10)
    plt.legend()
    plt.title(f"{title}")
    plt.xlabel("Date")
    plt.ylabel(label)
    plt.savefig(f"plots/{path_to_save}/{title}.png")
    plt.close()


def train_and_predict(title: str, model: str, path_to_data: str, y_col: str, x_col: str,
                      path_to_save: str):
    # load data
    df = pd.read_csv(path_to_data)
    df_list = np.array_split(df, 5)
    count = 0
    tittle_temp = title
    num_epochs = 50
    batch_size = 64
    # Define the sliding window size
    window_size = 24
    # iterate over each part to train and predict
    for data in df_list:
        # preprocess data
        if model == "lstm":
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            scaler = MinMaxScaler()
        X_train, X_test, y_train, y_test = _split_data_to_training_and_testing(model, data, y_col, x_col,
                                                                               window_size, scaler)
        test_data = data.iloc[-len(y_test):]

        # create LSTM model for wind speed prediction
        model = _create_model(model, X_train, y_train, window_size)

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Train the model
        model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)

        # make predictions on test data
        y_pred = model.predict(X_test)

        # inverse transform scaled data to original scale
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
        count += 1
        title = f"{tittle_temp} - Part {count}"
        _check_model_performance(path_to_save, title, y_test, y_pred)
        _plot_result(title, y_test, y_pred, path_to_save, test_data, x_col)


# # wind_speed tcn
# train_and_predict("Wind Speed Q1", "tcn", "datasets/splits/2007/Q1.csv", "wind speed", "wind speed", "tcn")
# train_and_predict("Wind Speed Q2", "tcn", "datasets/splits/2010/Q2.csv", "wind speed", "wind speed", "tcn")
# train_and_predict("Wind Speed Q3", "tcn", "datasets/splits/2010/Q3.csv", "wind speed", "wind speed", "tcn")
# train_and_predict("Wind Speed Q4", "tcn", "datasets/splits/2010/Q4.csv", "wind speed", "wind speed", "tcn")

# wind_direction tcn
train_and_predict("Wind direction Q1", "tcn", "datasets/splits/2010/Q1.csv", "wind direction", "wind direction", "tcn")
train_and_predict("Wind direction Q2", "tcn", "datasets/splits/2010/Q2.csv", "wind direction", "wind direction", "tcn")
train_and_predict("Wind direction Q3", "tcn", "datasets/splits/2010/Q3.csv", "wind direction", "wind direction", "tcn")
train_and_predict("Wind direction Q4", "tcn", "datasets/splits/2010/Q4.csv", "wind direction", "wind direction", "tcn")

# wind_speed lstm
# train_and_predict("Wind Speed Q1", "lstm", "datasets/splits/2007/Q1.csv", "wind speed", "wind speed", "lstm")
# train_and_predict("Wind Speed Q2", "lstm", "datasets/splits/2010/Q2.csv", "wind speed", "wind speed", "lstm")
# train_and_predict("Wind Speed Q3", "lstm", "datasets/splits/2010/Q3.csv", "wind speed", "wind speed", "lstm")
# train_and_predict("Wind Speed Q4", "lstm", "datasets/splits/2010/Q4.csv", "wind speed", "wind speed", "lstm")
#
# # wind_direction lstm
# train_and_predict("Wind direction Q1", "lstm", "datasets/splits/2010/Q1.csv", "wind direction", "wind direction",
#                   "lstm")
# train_and_predict("Wind direction Q2", "lstm", "datasets/splits/2010/Q2.csv", "wind direction", "wind direction",
#                   "lstm")
# train_and_predict("Wind direction Q3", "lstm", "datasets/splits/2010/Q3.csv", "wind direction", "wind direction",
#                  "lstm")
# train_and_predict("Wind direction Q4", "lstm", "datasets/splits/2010/Q4.csv", "wind direction", "wind direction",
 #                 "lstm")
