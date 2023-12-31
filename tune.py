import keras_tuner

from preprocess import generating_training_sequences
from preprocess import input_params, TESTING_DATASET_PATH, param_shapes
from train import SAVE_MODEL_PATH, BATCH_SIZE
from tensorflow import keras

TUNING_DIRECTORY = "tunings"
PROJECT_NAME = "CantoGem"


class PitchLoss(keras.layers.Layer):
    def _init_(self):
        super().__init__()

    def call(self, input_valid_pitches, output_pitch):
        # batch_size, last in sequence, whole range of pitches
        invalid_pitches = 1 - input_valid_pitches[:, -1, :]
        invalid_pitch_probabilities = keras.layers.Multiply()([invalid_pitches, output_pitch])
        # calculates the probability of invalid pitch averaged over batch size
        sum = keras.backend.sum(keras.backend.sum(invalid_pitch_probabilities)) * (1 / BATCH_SIZE)
        return keras.backend.log(sum * 5 + 1) * 2


def build_hyper_model(hp):
    inputs = dict()
    LSTMs = dict()

    outputs = dict()

    # create the model architecture
    for input_type in input_params:
        inputs[input_type] = keras.layers.Input(shape=(None, param_shapes[input_type]))
        tmp = keras.layers.LSTM(param_shapes[input_type], return_sequences=True)(inputs[input_type])
        LSTMs[input_type] = keras.layers.Dropout(hp.Float(f'input dropout: {input_type}', min_value=0.1,
                                                          max_value=0.9, step=0.1))(tmp)

    combined_input = keras.layers.concatenate(list(LSTMs.values()))

    combined_dropout_1 = hp.Float('combined input dropout 1', min_value=0.1, max_value=0.9, step=0.1)
    combined_dropout_2 = hp.Float('combined input dropout 2', min_value=0.1, max_value=0.9, step=0.1)
    combined_dropout_3 = hp.Float('combined input dropout 3', min_value=0.1, max_value=0.9, step=0.1)
    combined_dropout_4 = hp.Float('combined input dropout 4', min_value=0.1, max_value=0.9, step=0.1)
    combined_dropout_5 = hp.Float('combined input dropout 5', min_value=0.1, max_value=0.9, step=0.1)
    combined_dropout_6 = hp.Float('combined input dropout 5', min_value=0.1, max_value=0.9, step=0.1)
    combined_dropout_7 = hp.Float('combined input dropout 5', min_value=0.1, max_value=0.9, step=0.1)

    units_1 = hp.Int('Units 1', min_value=128, max_value=512, step=64)
    units_2 = hp.Int('Units 2', min_value=128, max_value=512, step=64)
    units_3 = hp.Int('Units 3', min_value=128, max_value=512, step=64)
    units_4 = hp.Int('Units 4', min_value=128, max_value=512, step=64)
    units_5 = hp.Int('Units 5', min_value=128, max_value=512, step=64)
    units_6 = hp.Int('Units 6', min_value=128, max_value=512, step=64)
    units_7 = hp.Int('Units 7', min_value=128, max_value=512, step=64)

    x = keras.layers.LSTM(units_1, return_sequences=True)(combined_input)
    x = keras.layers.Dropout(combined_dropout_1)(x)
    y = keras.layers.LSTM(units_2, return_sequences=True)(combined_input)
    y = keras.layers.Dropout(combined_dropout_2)(y)
    z = keras.layers.LSTM(units_3, return_sequences=True)(
        keras.layers.concatenate(
            [LSTMs["pos_internal"], LSTMs["current_tone"], LSTMs["next_tone"], LSTMs["future_words"]])
    )
    z = keras.layers.Dropout(combined_dropout_3)(z)

    x2 = keras.layers.LSTM(units_4)(keras.layers.concatenate([x, y, z]))
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.Dropout(combined_dropout_4)(x2)
    x = keras.layers.Dense(units_5, activation="relu")(x2)

    tmp = keras.layers.Dense(units_6, activation="relu")(x)
    tmp = keras.layers.BatchNormalization()(tmp)
    tmp = keras.layers.Dropout(combined_dropout_5)(tmp)
    outputs["pitch"] = keras.layers.Dense(param_shapes["pitch"], activation="softmax", name="pitch")(tmp)

    tmp = keras.layers.Dropout(combined_dropout_6)(outputs["pitch"])
    tmp = keras.layers.Dense(units_7, activation="relu")(keras.layers.concatenate([x, tmp]))
    tmp = keras.layers.BatchNormalization()(tmp)
    tmp = keras.layers.Dropout(combined_dropout_7)(tmp)
    outputs["duration"] = keras.layers.Dense(param_shapes["duration"], activation="softmax", name="duration")(tmp)

    lr = hp.Float('Learning rate', min_value=0.001, max_value=0.0025, step=0.0005)

    model = keras.Model(list(inputs.values()), list(outputs.values()))
    model.add_loss(PitchLoss()(inputs["valid_pitches"], outputs["pitch"]))

    # compile model
    model.compile(loss="categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=lr),
                  metrics=["accuracy"])

    return model


def tune():
    # generate the training sequences
    inputs, targets = generating_training_sequences()
    testing_inputs, testing_targets = generating_training_sequences(dataset_path=TESTING_DATASET_PATH)

    tuner = keras_tuner.BayesianOptimization(
        hypermodel=build_hyper_model,
        directory=TUNING_DIRECTORY,
        project_name=PROJECT_NAME,
        objective="val_loss",
        max_trials=20,
    )

    input_mapped = {}
    for i, key in enumerate(inputs):
        new_key = f"input_{i + 1}"
        input_mapped[new_key] = inputs[key]

    testing_input_mapped = {}
    for i, key in enumerate(testing_inputs):
        new_key = f"input_{i + 1}"
        testing_input_mapped[new_key] = testing_inputs[key]

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    print("-" * 20)
    tuner.search_space_summary()
    tuner.search(x=input_mapped, y=targets, epochs=40, validation_data=(testing_input_mapped, testing_targets),
                 callbacks=[early_stopping])
    print("-" * 20)
    tuner.results_summary()

    models = tuner.get_best_models(num_models=1)
    best_model = models[0]

    # Evaluate the model
    print("--- Model evaluation --- ")
    best_model.evaluate(x=list(testing_inputs.values()), y=list(testing_targets.values()))
    best_model.save_weights(SAVE_MODEL_PATH)


if __name__ == "__main__":
    tune()
