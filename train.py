from preprocess import generating_training_sequences
from preprocess import input_params, TESTING_DATASET_PATH, param_shapes, num_tone, num_phrasing
from tensorflow import keras

import matplotlib.pyplot as plt

from paths import SAVE_MODEL_PATH, PLOT_PATH

LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 50


class PitchLoss(keras.layers.Layer):
    def _init_(self):
        super().__init__()

    def call(self, input_valid_pitches, output_pitch):
        # batch_size, last in sequence, whole range of pitches
        invalid_pitches = 1 - input_valid_pitches[:, -1, :]
        invalid_pitch_probabilities = keras.layers.Multiply()([invalid_pitches, output_pitch])
        # calculates the probability of invalid pitch averaged over batch size
        sum = keras.backend.sum(keras.backend.sum(invalid_pitch_probabilities)) * (1 / BATCH_SIZE)
        return sum * sum * 2


INPUT_DROPOUTS = {
    "pitch": 0.2,
    "duration": 0.2,
    "pos_internal": 0.2,
    "pos_external": 0.2,
    "valid_pitches": 0.2,
    "current_tone": 0.2,
    "next_tone": 0.2,
    "when_end": 0.3,
    "when_rest": 0.3,
    "phrasing": 0.6,
    "future_words": 0.2,
}


def build_model():
    inputs = dict()
    LSTMs = dict()

    outputs = dict()

    # create the model architecture
    for type in input_params:
        inputs[type] = keras.layers.Input(shape=(None, param_shapes[type]))
        tmp = keras.layers.LSTM(param_shapes[type], return_sequences=True)(inputs[type])
        # slowly increase dropout the farther a tone is
        LSTMs[type] = keras.layers.Dropout(INPUT_DROPOUTS[type])(tmp)

    combined_input = keras.layers.concatenate(list(LSTMs.values()))

    x = keras.layers.LSTM(192, return_sequences=True)(combined_input)
    x = keras.layers.Dropout(0.4)(x)
    y = keras.layers.LSTM(128, return_sequences=True)(combined_input)
    y = keras.layers.Dropout(0.4)(y)
    z = keras.layers.LSTM(64, return_sequences=True)(
        keras.layers.concatenate([LSTMs["pos_internal"], LSTMs["current_tone"], LSTMs["next_tone"], LSTMs["future_words"]])
    )
    z = keras.layers.Dropout(0.4)(z)

    x2 = keras.layers.LSTM(384)(keras.layers.concatenate([x, y, z]))
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.Dropout(0.2)(x2)
    x = keras.layers.Dense(384, activation="relu")(x2)

    tmp = keras.layers.Dense(384, activation="relu")(x)
    tmp = keras.layers.BatchNormalization()(tmp)
    tmp = keras.layers.Dropout(0.4)(tmp)
    outputs["pitch"] = keras.layers.Dense(param_shapes["pitch"], activation="softmax", name="pitch")(tmp)

    tmp = keras.layers.Dropout(0.8)(outputs["pitch"])
    tmp = keras.layers.Dense(128, activation="relu")(keras.layers.concatenate([x, tmp]))
    tmp = keras.layers.BatchNormalization()(tmp)
    tmp = keras.layers.Dropout(0.7)(tmp)
    outputs["duration"] = keras.layers.Dense(param_shapes["duration"], activation="softmax", name="dur.")(tmp)

    model = keras.Model(list(inputs.values()), list(outputs.values()))
    model.add_loss(PitchLoss()(inputs["valid_pitches"], outputs["pitch"]))

    # compile model
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=["accuracy"])

    return model


def train():
    loadFromExist = input("Load model from existing? (Y/N) ").lower() == "y"
    print("Continuing training session." if loadFromExist else "Creating new model.")
    randomiseTrain = input("Randomise dataset weights? (Y/N) ").lower() == "y"
    print("Randomising weights." if randomiseTrain else "Inserting one of each song.")

    # generate the training sequences
    inputs, targets = generating_training_sequences(randomise_dataset=randomiseTrain)
    testing_inputs, testing_targets = generating_training_sequences(dataset_path=TESTING_DATASET_PATH)

    # build the network
    model = build_model()
    if loadFromExist:
        model.load_weights(SAVE_MODEL_PATH)

    # train the model
    # Create a callback that saves the model's weights
    cp_callback = [keras.callbacks.ModelCheckpoint(filepath=SAVE_MODEL_PATH, verbose=0, save_weights_only=True),
                   keras.callbacks.EarlyStopping(monitor="val_loss", patience=5000, verbose=0)]

    history = model.fit(list(inputs.values()), list(targets.values()), epochs=5000, batch_size=BATCH_SIZE,
                        callbacks=[cp_callback], validation_data=(testing_inputs.values(), testing_targets.values()))

    # Save the model
    model.save_weights(SAVE_MODEL_PATH)

    # Evaluate the model
    print("--- Model evaluation --- ")
    model.evaluate(x=list(testing_inputs.values()), y=list(testing_targets.values()))

    # Plot the model
    pitch_accuracies = history.history["pitch_accuracy"]
    duration_accuracies = history.history["dur._accuracy"]

    plt.plot(range(len(history.history['loss'])), pitch_accuracies, label="Pitch accuracy")
    plt.plot(range(len(history.history['loss'])), duration_accuracies, label="Duration accuracy")

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training accuracies")

    plt.show()
    plt.savefig(PLOT_PATH)


if __name__ == "__main__":
    train()
