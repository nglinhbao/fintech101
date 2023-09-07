from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, SimpleRNN, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os


# Create model function
def create_model(sequence_length, n_features, layer_name, units, n_layers=2, dropout=0.3,
                 loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    # Create a Sequential model
    model = Sequential()

    # Loop through the specified number of layers
    for i in range(n_layers):
        if i == 0:
            # First layer
            if bidirectional:
                # Add a Bidirectional layer if bidirectional is True
                model.add(Bidirectional(layer_name(units, return_sequences=True),
                                        batch_input_shape=(None, sequence_length, n_features)))
            else:
                # Add a single-directional layer_name layer
                model.add(layer_name(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # Last layer
            if bidirectional:
                # Add a Bidirectional layer for the last layer
                model.add(Bidirectional(layer_name(units, return_sequences=False)))
            else:
                # Add a single-directional layer_name layer for the last layer
                model.add(layer_name(units, return_sequences=False))
        else:
            # Hidden layers
            if bidirectional:
                # Add a Bidirectional layer for hidden layers
                model.add(Bidirectional(layer_name(units, return_sequences=True)))
            else:
                # Add a single-directional layer_name layer for hidden layers
                model.add(layer_name(units, return_sequences=True))

        # Add dropout after each layer
        model.add(Dropout(dropout))

    # Add a Dense layer with linear activation as the output layer
    model.add(Dense(1, activation="linear"))

    # Compile the model with specified loss, metrics, and optimizer
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)

    # Return the created model
    return model


def train_model(model, model_name, df, batch_size, epochs):
    # some tensorflow callbacks
    checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True,
                                   save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    # train the model and save the weights whenever we see
    # a new optimal model using ModelCheckpoint
    history = model.fit(df["X_train"], df["y_train"],
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(df["X_test"], df["y_test"]),
                        callbacks=[checkpointer, tensorboard],
                        verbose=1)
