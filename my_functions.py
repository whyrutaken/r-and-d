from datetime import datetime
# from keras.utils.vis_utils import plot_model
from pathlib2 import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.python.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
import my_models as mm

input_file = 'input/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv'


# Creates the folders which will be used later on
def create_path(model_number):
    date = datetime.now().strftime("%x")
    date = date.replace("/", "-")

    # define path names
    path = date + "/model" + model_number
    log_path = path + "/log"
    loss_path = path + "/loss_plots"
    pred_path = path + "/prediction_plots"
    model_plot_path = path + "/model_plots"
    saved_models_path = path + "/saved_models"
    best_params_path = path + "/best_params"

    # create path names
    Path(path).mkdir(parents=True, exist_ok=True)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    Path(loss_path).mkdir(parents=True, exist_ok=True)
    Path(pred_path).mkdir(parents=True, exist_ok=True)
    Path(model_plot_path).mkdir(parents=True, exist_ok=True)
    Path(saved_models_path).mkdir(parents=True, exist_ok=True)
    Path(best_params_path).mkdir(parents=True, exist_ok=True)

    return path, pred_path


# Saves results of the models into file
def save_results(scores, predictions, duration_time, path, data_scaling):
    predictions = np.array(predictions)
    scores.to_csv(path + "/" + data_scaling + "_scores.csv")
    np.save(path + "/" + data_scaling + "_predictions.npy", predictions)

    with open(path + "/" + data_scaling + "_duration_time.txt", "w") as f:
        f.write(str(duration_time))


# Preprocesses the data and splits it into train and test sets
def preprocess_and_split_data():
    ## PREPROCESS

    # Load in dataset
    df = pd.read_csv(input_file)
    # Set values of "Timestamp" to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    # Set Timestamp as index
    df = df.set_index('Timestamp')
    df.sort_index(inplace=True)

    # Using the last 4 years (2.102.400 minutes)
    # from 2017-04-01 to 2021-03-31
    # 365*24*60*4
    df = df.iloc[-365 * 24 * 60 * 4:, :]

    # Drop rows with NaN values
    # Remains: 2.038.838
    df.dropna(inplace=True)

    ## TRAIN-TEST SPLIT

    ## Split with keras.train_test_split
    df_target = df["Weighted_Price"]
    del df["Weighted_Price"]
    x_train, x_test, y_train, y_test = train_test_split(df, df_target, test_size=0.3, shuffle=False)
    y_train, y_test = pd.DataFrame(y_train), pd.DataFrame(y_test)

    ## Split to train and test sets (without keras)
    # split = int(0.7*len(df))
    # train_df, test_df = df[:split], df[split:]

    return x_train, x_test, y_train, y_test


# Prepares sliding windows
def prepare_sliding_windows(feature, target, sliding_window=16):
    X, Y = [], []
    for i in range(len(feature) - sliding_window):
        X.append(feature[i:(i + sliding_window), :])  # features
        Y.append(target[i + sliding_window, -1])  # target value (weighted_price)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


# Scales the data on dataset level and prepares sliding windows
def sliding_windows_and_scale_DL(x_train, x_test, y_train, y_test, sliding_window=16):
    ## scaling
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_scale = x_train.append(x_test)
    y_scale = y_train.append(y_test)

    scaler_x.fit(x_scale)
    scaler_y.fit(y_scale)

    x_train_scaled = scaler_x.transform(x_train)
    x_test_scaled = scaler_x.transform(x_test)
    y_train_scaled = scaler_y.transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    ## preparing sliding windows
    x_train_w, y_train_w = prepare_sliding_windows(x_train_scaled, y_train_scaled, sliding_window)
    x_test_w, y_test_w = prepare_sliding_windows(x_test_scaled, y_test_scaled, sliding_window)

    return x_train_w, x_test_w, y_train_w, y_test_w, scaler_y


# Prepares sliding windows then scales the data on a sliding window level
def sliding_windows_and_scale_WL(x_train, x_test, y_train, y_test, sliding_window=16):
    ## preparing sliding windows

    x_train_w, y_train_w = prepare_sliding_windows(x_train.to_numpy(), y_train.to_numpy(), sliding_window)
    x_test_w, y_test_w = prepare_sliding_windows(x_test.to_numpy(), y_test.to_numpy(), sliding_window)

    ## scaling
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    y_scale = y_train.append(y_test)
    scaler_y.fit(y_scale)

    x_train_scaled = x_train_w
    for i in range(len(x_train_scaled)):
        scaler_x.fit(x_train_scaled[i])
        x_train_scaled[i] = scaler_x.transform(x_train_scaled[i])

    y_train_scaled = np.reshape(y_train_w, (-1, 1))
    y_train_scaled = scaler_y.transform(y_train_scaled)

    x_test_scaled = x_test_w
    for i in range(len(x_test_scaled)):
        scaler_x.fit(x_test_scaled[i])
        x_test_scaled[i] = scaler_x.transform(x_test_scaled[i])

    y_test_scaled = np.reshape(y_test_w, (-1, 1))
    y_test_scaled = scaler_y.transform(y_test_scaled)

    return x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, scaler_y


# Set callback for the models
def set_callbacks(path, data_scaling, sliding_window, batch_size, epoch):
    early_stopper = EarlyStopping(monitor='loss', patience=20, mode="min", restore_best_weights=True, verbose=2)
    csv_logger = 0
    if data_scaling == "DL":
        csv_logger = CSVLogger(
            path + "/log/dl_training" + str(sliding_window) + "-" + str(epoch) + "-" + str(batch_size) + ".log",
            separator=',', append=False)
    if data_scaling == "WL":
        csv_logger = CSVLogger(
            path + "/log/wl_training" + str(sliding_window) + "-" + str(epoch) + "-" + str(batch_size) + ".log",
            separator=',', append=False)

    return [early_stopper, csv_logger]


def grid_search_lstm(x_train_scaled, y_train_scaled, build_model):
    dropout_rate_opts = (0, 0.3, 0.6)
    hidden_layers_opts = (16, 32, 64)

    # n_timesteps, n_features
    input_shape = (x_train_scaled.shape[1], x_train_scaled.shape[2])

    model = KerasRegressor(
        build_fn=build_model,
        hidden_layer=hidden_layers_opts, dropout=dropout_rate_opts, input_shape=input_shape
    )

    # epochs=epochs, batch_size=batches,
    hyperparameters = dict(hidden_layer=hidden_layers_opts, dropout=dropout_rate_opts)
    # get rid of the cross-validation
    # source: https://stackoverflow.com/questions/44636370/scikit-learn-gridsearchcv-without-cross-validation-unsupervised-learning
    cv = [(slice(None), slice(None))]

    rs = GridSearchCV(model, param_grid=hyperparameters, cv=cv, n_jobs=-1)
    rs.fit(x_train_scaled, y_train_scaled, verbose=1)
    best_params = rs.best_params_

    model = build_model(best_params["hidden_layer"], best_params["dropout"], input_shape)

    return model, best_params


def grid_search_cnn(x_train_scaled, y_train_scaled, build_model):
    filter_opts = (16, 32, 64)
    kernel_opts = (1, 3, 5)
    dropout_opts = (0, 0.3, 0.6)

    # n_timesteps, n_features
    input_shape = (x_train_scaled.shape[1], x_train_scaled.shape[2])

    model = KerasRegressor(
        build_fn=build_model,
        filters=filter_opts, kernel_size=kernel_opts, dropout=dropout_opts, input_shape=input_shape
    )

    # epochs=epochs, batch_size=batches,
    hyperparameters = dict(filters=filter_opts, kernel_size=kernel_opts, dropout=dropout_opts)

    # get rid of the cross-validation
    # source: https://stackoverflow.com/questions/44636370/scikit-learn-gridsearchcv-without-cross-validation-unsupervised-learning
    cv = [(slice(None), slice(None))]

    rs = GridSearchCV(model, param_grid=hyperparameters, cv=cv, n_jobs=-1)
    rs.fit(x_train_scaled, y_train_scaled, verbose=1)
    best_params = rs.best_params_

    model = build_model(best_params["filters"], best_params["kernel_size"], best_params["dropout"], input_shape)

    return model, best_params


def grid_search_cnnlstm(x_train_scaled, y_train_scaled, build_model):
    filter_opts = (16, 32, 64)
    kernel_opts = (2, 3, 5)
    hidden_layer_opts = (16, 32, 64)

    # n_timesteps, n_features
    input_shape = (2, int(x_train_scaled.shape[1] / 2), x_train_scaled.shape[2])

    model = KerasRegressor(
        build_fn=build_model,
        filters=filter_opts, kernel_size=kernel_opts, hidden_layer=hidden_layer_opts, input_shape=input_shape
    )

    # epochs=epochs, batch_size=batches,
    hyperparameters = dict(filters=filter_opts, kernel_size=kernel_opts, hidden_layer=hidden_layer_opts)

    # get rid of the cross-validation
    # source: https://stackoverflow.com/questions/44636370/scikit-learn-gridsearchcv-without-cross-validation-unsupervised-learning
    cv = [(slice(None), slice(None))]

    rs = GridSearchCV(model, param_grid=hyperparameters, cv=cv, n_jobs=-1)
    rs.fit(x_train_scaled, y_train_scaled, verbose=1)
    best_params = rs.best_params_

    model = build_model(best_params["filters"], best_params["kernel_size"], best_params["hidden_layer"], input_shape)

    return model, best_params


def grid_search_convlstm(x_train_scaled, y_train_scaled, build_model):
    filter_opts = (16, 32, 64)
    dropout_opts = (0, 0.3, 0.6)

    # n_timesteps, n_features
    input_shape = (x_train_scaled.shape[1], x_train_scaled.shape[2], x_train_scaled.shape[3], x_train_scaled.shape[4])
    model = KerasRegressor(
        build_fn=build_model,
        filters=filter_opts, dropout=dropout_opts, input_shape=input_shape
    )

    # epochs=epochs, batch_size=batches,
    hyperparameters = dict(filters=filter_opts, dropout=dropout_opts)

    # get rid of the cross-validation
    # source: https://stackoverflow.com/questions/44636370/scikit-learn-gridsearchcv-without-cross-validation-unsupervised-learning
    cv = [(slice(None), slice(None))]

    rs = GridSearchCV(model, param_grid=hyperparameters, cv=cv, n_jobs=-1)
    rs.fit(x_train_scaled, y_train_scaled, verbose=1)
    best_params = rs.best_params_

    model = build_model(best_params["filters"], best_params["dropout"], input_shape)

    return model, best_params


def evaluate_model(sliding_windows, epochs, batches, model_number="1", data_scaling="DL", path=""):
    start = datetime.now()
    training_scores = []
    predictions = []

    for s in sliding_windows:
        for e in epochs:
            for b in batches:
                model, prediction, overall_score = fit_model(model_number, data_scaling, path, sliding_window=s,
                                                             batch_size=b, epoch=e)
                experiment = [s, e, b]
                temp = experiment.copy()
                temp.append(prediction)
                predictions.append(temp)
                experiment.append(overall_score)
                training_scores.append(experiment)
                end = datetime.now()
                print("Duration: {}".format(end - start))
    end = datetime.now()
    duration = end - start
    print("Total duration: {}".format(duration))

    df_training_scores = pd.DataFrame(training_scores, columns=['SW', 'Epoch', 'Batches', 'RMSE'])

    return df_training_scores, predictions, duration


def fit_model(model_number, data_scaling="DL", path="", sliding_window=16,
              batch_size=32, epoch=10):
    # Get and preprocess data
    x_train, x_test, y_train, y_test = preprocess_and_split_data()

    # Initialize variables
    x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = 0, 0, 0, 0
    scaler_y = 0

    ## SCALE on Dataset Level and prepare SLIDING WINDOWS
    if data_scaling == "DL":
        x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, scaler_y = sliding_windows_and_scale_DL(
            x_train, x_test, y_train, y_test, sliding_window)
    ## prepare SLIDING WINDOWS and SCALE on Window Level
    if data_scaling == "WL":
        x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, scaler_y = sliding_windows_and_scale_WL(
            x_train, x_test, y_train, y_test, sliding_window)

    n_steps = 2
    n_length = int(sliding_window / n_steps)
    n_timesteps = x_train_scaled.shape[1]
    n_features = x_train_scaled.shape[2]
    n_outputs = 1

    # if CNN-LSTM, reshape x_train and x_test
    if (int(model_number) > 10 and int(model_number) <= 15):
        x_train_scaled = x_train_scaled.reshape((x_train_scaled.shape[0], n_steps, n_length, n_features))
        x_test_scaled = x_test_scaled.reshape((x_test_scaled.shape[0], n_steps, n_length, n_features))

    # if ConvLSTM, reshape x_train and x_test
    if int(model_number) > 15:
        x_train_scaled = x_train_scaled.reshape((x_train_scaled.shape[0], n_steps, 1, n_length, n_features))
        x_test_scaled = x_test_scaled.reshape((x_test_scaled.shape[0], n_steps, 1, n_length, n_features))

    # set Callbacks
    my_callbacks = set_callbacks(path, data_scaling, sliding_window, batch_size, epoch)

    # Select model
    model = 0
    best_params = 0
    if model_number == "1":
        model = mm.lstm1(n_timesteps, n_features)
    if model_number == "2":
        model = mm.lstm2(n_timesteps, n_features)
    if model_number == "3":
        model, best_params = grid_search_lstm(x_train_scaled, y_train_scaled, mm.lstm3)
    if model_number == "4":
        model, best_params = grid_search_lstm(x_train_scaled, y_train_scaled, mm.lstm4)
    if model_number == "5":
        model, best_params = grid_search_lstm(x_train_scaled, y_train_scaled, mm.lstm5)
    if model_number == "6":
        model = mm.cnn1(n_timesteps, n_features)
    if model_number == "7":
        model = mm.cnn2(n_timesteps, n_features)
    if model_number == "8":
        model, best_params = grid_search_cnn(x_train_scaled, y_train_scaled, mm.cnn3)
    if model_number == "9":
        model, best_params = grid_search_cnn(x_train_scaled, y_train_scaled, mm.cnn4)
    if model_number == "10":
        model, best_params = grid_search_cnn(x_train_scaled, y_train_scaled, mm.cnn5)
    if model_number == "11":
        model = mm.cnn_lstm1(n_length, n_features)
    if model_number == "12":
        model = mm.cnn_lstm2(n_length, n_features)
    if model_number == "13":
        model = mm.cnn_lstm3(n_length, n_features)
    if model_number == "14":
        model = mm.cnn_lstm4(n_length, n_features)
    if model_number == "15":
        #    model, best_params = grid_search_cnnlstm(x_train_scaled, y_train_scaled, mm.cnn_lstm5)
        model = mm.cnn_lstm5(n_length, n_features)
    if model_number == "16":
        model = mm.conv_lstm1(n_outputs, n_steps, n_length, n_features)
    if model_number == "17":
        model = mm.conv_lstm2(n_outputs, n_steps, n_length, n_features)
    if model_number == "18":
        model, best_params = grid_search_convlstm(x_train_scaled, y_train_scaled, mm.conv_lstm3)
    if model_number == "19":
        model, best_params = grid_search_convlstm(x_train_scaled, y_train_scaled, mm.conv_lstm4)
    if model_number == "20":
        model, best_params = grid_search_convlstm(x_train_scaled, y_train_scaled, mm.conv_lstm5)

    # Plot model's architecture
    #   plot_model(model,
    #              to_file=path + "/model_plots/model_plot" + str(sliding_window) + "-" + str(batch_size) + "-" + str(
    #                  epoch) + ".png", show_shapes=True, show_layer_names=True)

    # Fit model
    history = model.fit(x_train_scaled, y_train_scaled, epochs=epoch, batch_size=batch_size, validation_split=0.15,
                        verbose=1, shuffle=False, callbacks=my_callbacks)
    # Plot losses of model
    plot_loss(history, (path + "/loss_plots"), data_scaling, sliding_window, batch_size, epoch)
    model.save(
        path + "/saved_models/" + data_scaling + "_saved_model" + str(sliding_window) + "-" + str(epoch) + "-" + str(
            batch_size))

    # save chosen result of GridSearch
    if best_params != 0:
        with open(path + '/best_params/' + data_scaling + '_best_params' + str(sliding_window) + "-" + str(
                epoch) + "-" + str(
            batch_size) + '.txt', 'w') as f:
            print(best_params, file=f)

    # Predict model
    prediction = predict_model(model, x_test_scaled, scaler_y)
    overall_score = evaluate_forecasts(y_test, prediction)
    return model, prediction, overall_score


def predict_model(model, x_test_scaled, scaler_y):
    y_pred = model.predict(x_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred)
    return y_pred


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    actual = actual.iloc[:len(predicted)]
    actual = actual.to_numpy()
    # calculate a MSE score for each minute
    mse = tf.keras.metrics.mean_squared_error(actual, predicted).numpy()

    # overall RMSE
    MSE = mse.sum() / len(actual)
    RMSE = np.sqrt(MSE)

    return RMSE


## --- PLOT RESULTS

def plot_loss(history, path, data_scaling="DL", sliding_window=16, batch_size=32, nb_epoch=10):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training loss', 'Validation loss'])
    if data_scaling == "DL":
        plt.savefig(path + "/dl_loss" + str(sliding_window) + "-" + str(nb_epoch) + "-" + str(batch_size) + ".png")
    if data_scaling == "WL":
        plt.savefig(path + "/wl_loss" + str(sliding_window) + "-" + str(nb_epoch) + "-" + str(batch_size) + ".png")
#  plt.show()


def plot_predictions(predictions, actual, path, data_scaling="DL"):
    for i in range(len(predictions)):
        y_pred = predictions[i][3]
        len_ = len(y_pred)
        y_index = list(range(len(y_pred)))
        plt.figure()
        plt.plot(y_index, actual[:len_], color='red')
        plt.plot(y_index, y_pred[:])
        plt.legend(["Original", "Prediction"])
        plt.savefig(
            path + "/" + data_scaling + "_pred" + str(predictions[i][0]) + "-" + str(predictions[i][1]) + "-" + str(
                predictions[i][2]) + ".png")


#      if data_scaling == "WL":
#         plt.savefig(path + "/wl_pred" + str(predictions[i][0]) + "-" + str(predictions[i][1]) + "-" + str(
#            predictions[i][2]) + ".png")
#    plt.show()


def plot_metrics(history):
    plt.figure()
    plt.plot(history.history['mse'])
    plt.plot(history.history['mae'])
    plt.plot(history.history['mape'])
    plt.legend(['mse', 'mae', 'mape'])
    plt.legend(['mape'])
    plt.show()


# Plot history and future
def plot_multistep(y_train, y_pred_, y_test):
    plt.figure(figsize=(20, 4))
    y_mean = np.mean(y_pred_)
    range_history = len(y_train)
    range_future = list(range(range_history, range_history + len(y_pred_)))
    # range_future = np.concatenate(y_train.index, y_test.index)
    plt.plot(np.arange(range_history), np.array(y_train), label='Train Data')
    plt.plot(range_future, np.array(y_pred_), label='Forecasted with LSTM')
    plt.plot(range_future, np.array(y_test), label='Test Data')
    plt.legend(loc='lower right')
    plt.title("Test Data", fontsize=18)
    plt.xlabel('Time step', fontsize=18)
    plt.ylabel('y-value', fontsize=18)
    plt.show()


