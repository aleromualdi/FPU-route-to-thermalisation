from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.models import Sequential


def autoencoder_model(timesteps, n_features=64):
    sequential = Sequential()
    sequential.add(
        LSTM(
            32,
            activation="relu",
            input_shape=(timesteps, n_features),
            return_sequences=True,
        )
    )
    # sequential.add(LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=regularizers.L1(10e-)))
    sequential.add(LSTM(8, activation="relu", return_sequences=False))
    sequential.add(RepeatVector(timesteps))
    sequential.add(LSTM(8, activation="relu", return_sequences=True))
    sequential.add(LSTM(32, activation="relu", return_sequences=True))
    sequential.add(TimeDistributed(Dense(n_features)))
    sequential.compile(optimizer="adam", loss="mse")
    return sequential
