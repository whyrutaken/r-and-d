from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import ConvLSTM2D, TimeDistributed
from tensorflow.python.keras.layers import Dropout, AveragePooling1D, BatchNormalization


## MODEL 1-5: LSTM
def lstm1(n_timesteps, n_features):
    model = Sequential()
    model.add(LSTM(64, input_shape=(n_timesteps, n_features), activation="relu",
                   return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dense(16))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def lstm2(n_timesteps, n_features):
    model = Sequential()
    model.add(LSTM(16, input_shape=(n_timesteps, n_features), activation="relu",
                   return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def lstm3(hidden_layer, dropout, input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, activation="relu",
                   return_sequences=True, dropout=dropout))
    model.add(LSTM(32, activation='relu', return_sequences=False, dropout=dropout))
    model.add(Dense(16))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def lstm4(hidden_layer, dropout, input_shape):
    model = Sequential()
    model.add(LSTM(16, input_shape=input_shape, activation="relu",
                   return_sequences=True, dropout=dropout))
    model.add(LSTM(32, activation='relu', return_sequences=True, dropout=dropout))
    model.add(LSTM(32, activation='relu', return_sequences=True, dropout=dropout))
    model.add(LSTM(64, activation='relu', return_sequences=False, dropout=dropout))
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def lstm5(hidden_layer, dropout, input_shape):
    model = Sequential()
    model.add(LSTM(hidden_layer, input_shape=input_shape, activation='relu',
                   return_sequences=False, dropout=dropout))
    model.add(Dense(hidden_layer))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


## MODEL 6-10: CNN
def cnn1(n_timesteps, n_features):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                     input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def cnn2(n_timesteps, n_features):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                     input_shape=(n_timesteps, n_features)))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model



def cnn3(filters, kernel_size, dropout, input_shape):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def cnn4(filters, kernel_size, dropout, input_shape):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
                     input_shape=input_shape))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def cnn5(filters, kernel_size, dropout, input_shape):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


# MODEL 11-15: CNN + LSTM
def cnn_lstm1(n_timesteps, n_features):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=2, activation='relu'),
                              input_shape=(2, n_timesteps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, return_sequences=True, activation="relu"))
    model.add(LSTM(32, return_sequences=False, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def cnn_lstm2(n_timesteps, n_features):
    model = Sequential()
    model.add(
        TimeDistributed(Conv1D(filters=32, kernel_size=2, activation='relu'),
                        input_shape=(2, n_timesteps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32, return_sequences=False, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def cnn_lstm3(n_timesteps, n_features):
    model = Sequential()
    model.add(
        TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'),
                        input_shape=(2, n_timesteps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32, return_sequences=False, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def cnn_lstm4(n_timesteps, n_features):
    model = Sequential()
    model.add(
        TimeDistributed(Conv1D(filters=32, kernel_size=2, activation='relu'),
                        input_shape=(2, n_timesteps, n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation="relu")))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32, return_sequences=False, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def cnn_lstm5(n_timesteps, n_features):
    model = Sequential()
    model.add(
        TimeDistributed(Conv1D(filters=32, kernel_size=2, activation='relu'),
                        input_shape=(2, n_timesteps, n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation="relu")))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, return_sequences=True, activation="relu"))
    model.add(LSTM(32, return_sequences=False, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


# MODEL 16-20: ConvLSTM
def conv_lstm1(n_outputs, n_steps, n_length, n_features):
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3),
                         activation='relu',
                         input_shape=(n_steps, 1, n_length, n_features),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=32, kernel_size=(1, 3),
                         activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(16))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def conv_lstm2(n_outputs, n_steps, n_length, n_features):
    model = Sequential()
    model.add(ConvLSTM2D(filters=16, kernel_size=(1, 3), activation='relu',
                   input_shape=(n_steps, 1, n_length, n_features),
                   padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=32, kernel_size=(1, 3), activation='relu',
                   padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=32, kernel_size=(1, 3), activation='relu',
                   padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu',
                   padding="same"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def conv_lstm3(filters, dropout, input_shape):
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu',
                   input_shape=input_shape, padding='same',
                   return_sequences=True, dropout=dropout))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=32, kernel_size=(1, 3), activation='relu',
                         padding='same', dropout=dropout))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(16))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def conv_lstm4(filters, dropout, input_shape):
    model = Sequential()
    model.add(ConvLSTM2D(filters=16, kernel_size=(1, 3), activation='relu',
                   input_shape=input_shape, padding='same',
                   return_sequences=True, dropout=dropout))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=32, kernel_size=(1, 3), activation='relu',
                         padding='same', return_sequences=True, dropout=dropout))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=32, kernel_size=(1, 3), activation='relu',
                         padding='same', return_sequences=True, dropout=dropout))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='relu',
                         padding="same", dropout=dropout))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model


def conv_lstm5(filters, dropout, input_shape):
    model = Sequential()
    model.add(
        ConvLSTM2D(filters=filters, kernel_size=(1, 3), activation='relu',
                   input_shape=input_shape, padding="same",
                   dropout=dropout))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
    model.summary()
    return model
