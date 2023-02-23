import pandas as pd
import numpy as np
from keras import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt


def train_and_predict(title: str, path_to_data: str, x_col, y_col, path_to_save: str):
    # Load the wind speed data
    data = pd.read_csv(path_to_data, parse_dates=[x_col], index_col=x_col)
    y_col_reshape = data[y_col].values.reshape(-1, 1)

    # Perform data preprocessing
    scaler = MinMaxScaler()
    wind_speed = scaler.fit_transform(y_col_reshape)

    # Split the data into training and test sets
    train_size = int(len(wind_speed) * 0.8)
    train_data, test_data = wind_speed[:train_size], wind_speed[train_size:]

    # Define the sliding window size
    window_size = 24

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

    # Define the TCN model
    model = Sequential([
        TCN(input_shape=(window_size, 1), nb_filters=64, kernel_size=3, dropout_rate=0.2),
        Flatten(),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    num_epochs = 50
    batch_size = 64
    # Train the model
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Open a file in append mode
    with open("performance.txt", "a") as f:
        # Write some text to the file
        f.write(f"---------------TCM Model Performance for {title}---------------\n")
        f.write(f"{title} MSE: {mse}\n")
        f.write(f"{title} MAE: {mae}\n")
        f.write(f"{title} RMSE: {rmse}\n")
        f.write("----------------------------------------------------------------\n")

    print(f"{title} MSE:", mse)
    print(f"{title} MAE:", mae)
    print(f"{title} RMSE:", rmse)

    # Plot the predicted and true wind speed values
    plt.plot(y_test, label=f'Actual {title}')
    plt.plot(y_pred, label=f'Predicted {title}')
    plt.legend()
    plt.title(f"{title}")
    plt.savefig(f"plots/{path_to_save}/{title}.png")
    plt.close()

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
