import os
import json
import re
import random
from sys import stdout

import music21 as m21
import numpy as np
import pycantonese as pc

RAW_DATA_PATH = "rawdata"
RAW_TESTING_DATA_PATH = "rawdata_testing"
DATASET_PATH = "dataset"
TESTING_DATASET_PATH = "testing_dataset"
MAPPING_PATH = "mappings"

SEQUENCE_LENGTH = 64

REST_TONE = 7
LONG_REST_TONE = 8
PAD_TONE = 0
END_TONE = 9
SEPARATOR_TONE = 10

PAD_NOTE = {"tone": PAD_TONE, "pitch": 1, "duration": 0, "phrasing": 0}
separator_note = {"tone": SEPARATOR_TONE, "pitch": 1, "duration": 0, "phrasing": 5}

ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2,
    2.5,
    3,
    3.5,
    4
]
PITCH_LOWER_LIMIT = 55  # G3
PITCH_UPPER_LIMIT = 79

with open(os.path.join(MAPPING_PATH, "pitch_mapping.json"), "r") as pitch_file:
    pitch_to_id = json.load(pitch_file)

with open(os.path.join(MAPPING_PATH, "duration_mapping.json"), "r") as duration_file:
    duration_to_id = json.load(duration_file)

with open("intelligible_intervals.json", "r") as interval_file:
    tones_to_intervals = json.load(interval_file)

id_to_pitch = {v: int(k) for k, v in pitch_to_id.items()}
id_to_duration = {v: int(k) for k, v in duration_to_id.items()}

num_pitch = len(pitch_to_id)
num_duration = len(duration_to_id)
num_tone = 11
num_pos_internal = 15
num_pos_external = 4
num_phrasing = 6
num_when_end = 64
num_when_rest = 8

input_params = ("pitch", "duration", "pos_internal", "pos_external", "valid_pitches", "phrasing", "current_tone",
                "next_tone", "when_end", "when_rest", "future_words")
inputs_to_treat_specially = ("future_words")
output_params = ("pitch", "duration")

param_shapes = {
    "pitch": num_pitch,
    "duration": num_duration,
    "pos_internal": num_pos_internal,  # Note position within a single bar
    "pos_external": num_pos_external,  # Note position within 4-bar phrase
    "valid_pitches": num_pitch,
    "phrasing": num_phrasing,
    "current_tone": num_tone,
    "next_tone": num_tone,
    "when_end": num_when_end,
    "when_rest": num_when_rest,
    "future_words": num_phrasing + num_tone
}


def create_datasets_and_mapping(raw_data_paths, save_dirs):
    # load the songs
    print("Loading songs...")
    encoded_songs_combined = []

    for path_idx, raw_data_path in enumerate(raw_data_paths):
        encoded_songs = load_songs(raw_data_path)
        num_songs = len(encoded_songs)

        print(f"Loaded {num_songs} songs.")

        for i, encoded_song in enumerate(encoded_songs):
            encoded_songs_combined += encoded_song

            with open(os.path.join(save_dirs[path_idx], str(i) + ".json"), "w") as fp:
                json.dump(encoded_song, fp, indent=4)

    create_mapping(encoded_songs_combined)


def load_songs(dataset_path):
    songs = []
    for path, subdir, files in os.walk(dataset_path):
        for sub in subdir:  # loop over subdirectories in path
            sub_path = os.path.join(path, sub)  # get full subdirectory path
            for file in os.listdir(sub_path):
                if file[-3:] == "mxl":
                    file_path = os.path.join(sub_path, file)  # get full file path
                    song = m21.converter.parse(file_path)
                    if not is_acceptable(song):
                        continue
                    print(file)
                    song = transpose(song)
                    song = encode_song(song)
                    songs.append(song)
    return songs


def is_acceptable(song):
    lowest_note = 1000
    highest_note = 0
    for note in song.flat.notesAndRests:
        if isinstance(note, m21.note.Note):
            lowest_note = min(lowest_note, note.pitch.midi)
            highest_note = max(highest_note, note.pitch.midi)
        if note.duration.quarterLength not in ACCEPTABLE_DURATIONS:
            return False

    if highest_note - lowest_note > PITCH_UPPER_LIMIT - PITCH_LOWER_LIMIT:
        return False
    return True


def transpose(song):
    # get key from song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # get interval for transposition
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    transposed_song = song.transpose(interval)

    lowest_note = 1000
    highest_note = 0
    for event in transposed_song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            lowest_note = min(lowest_note, event.pitch.midi)
            highest_note = max(highest_note, event.pitch.midi)

    if lowest_note < PITCH_LOWER_LIMIT:
        transposed_song = transposed_song.transpose(12)
    elif lowest_note > PITCH_UPPER_LIMIT:
        transposed_song = transposed_song.transpose(-12)

    return transposed_song


def encode_song(song, time_step=0.25):
    encoded_song = []
    all_tones = convert_song_to_lyric_data(song)
    current_note = {"pitch": None, "duration": None, "tone": None, "phrasing": None}

    num_notes = 0
    num_unintelligible_intervals = 0

    for event in song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            if current_note["pitch"] is None:
                current_note.update({"pitch": event.pitch.midi, "duration": event.duration.quarterLength,
                                     "tone": all_tones[0]["tone"], "phrasing": all_tones[0]["phrasing"]})
                all_tones.pop(0)
                continue

            elif current_note["pitch"] == event.pitch.midi and event.tie is not None and not event.tie.type == "start":
                current_note["duration"] += event.duration.quarterLength
                continue
            else:
                encoded_song.append(
                    {"pitch": current_note["pitch"], "duration": int(current_note["duration"] / time_step),
                     "tone": current_note["tone"], "phrasing": current_note["phrasing"]})

                last_note = {key: value for key, value in current_note.items()}

                current_note.update({"pitch": event.pitch.midi, "duration": event.duration.quarterLength,
                                     "tone": all_tones[0]["tone"], "phrasing": all_tones[0]["phrasing"]})
                all_tones.pop(0)

                # Check unintelligible intervals
                num_notes += 1
                key = str(last_note["tone"]) + "_" + str(current_note["tone"])
                if key in tones_to_intervals:
                    interval = current_note["pitch"] - last_note["pitch"]
                    if interval not in tones_to_intervals[key]:
                        num_unintelligible_intervals += 1
                        encoded_song.append(separator_note)

        elif isinstance(event, m21.note.Rest):
            if current_note["pitch"] is None:
                current_note.update({"pitch": 0, "duration": event.duration.quarterLength,
                                     "tone": LONG_REST_TONE if event.duration.quarterLength >= 2 else REST_TONE,
                                     "phrasing": 0})
                continue

            # Merge two rests
            if current_note["pitch"] == 0:
                current_note["duration"] += event.duration.quarterLength
                current_note["tone"] = LONG_REST_TONE if current_note["duration"] >= 2 else REST_TONE
                current_note["phrasing"] = 0
                continue
            else:
                encoded_song.append(
                    {"pitch": current_note["pitch"], "duration": int(current_note["duration"] / time_step),
                     "tone": current_note["tone"], "phrasing": current_note["phrasing"]})

            current_note.update({"pitch": 0, "duration": event.duration.quarterLength,
                                 "tone": LONG_REST_TONE if event.duration.quarterLength >= 2 else REST_TONE,
                                 "phrasing": 0})

    # Append the last note
    if current_note["pitch"] is not None:
        encoded_song.append({"pitch": current_note["pitch"], "duration": int(current_note["duration"] / time_step),
                             "tone": current_note["tone"], "phrasing": current_note["phrasing"]})
    
    while encoded_song[-1]["pitch"] < 10:
        encoded_song.pop()

    unintelligible_percentage = (num_unintelligible_intervals / num_notes) * 100
    print(f"Unintelligible: {unintelligible_percentage:.2f}%")
    return encoded_song


def convert_song_to_lyric_data(song):
    all_tones = []
    all_words = ""

    for i, event in enumerate(song.flat.notesAndRests):
        if not isinstance(event, m21.note.Note):
            continue

        if event.tie is not None and not event.tie.type == "start":
            continue

        if event.lyric is None:
            print(song.flat.notesAndRests[i-2].lyric + song.flat.notesAndRests[i-1].lyric)
            raise AttributeError("Lyrics cannot be null")
        all_words += event.lyric

    tokens = pc.parse_text(all_words).tokens()

    for token in tokens:
        anglicised_words = re.split(r'(?<=[0-9])(?=[a-zA-Z])', token.jyutping)
        tones = [{"tone": int(char[-1]), "phrasing": len(anglicised_words) - idx} for idx, char in
                 enumerate(anglicised_words)]
        all_tones.extend(tones)

    return all_tones


def create_mapping(encoded_song):
    # "pitch": 1 and "duration": 0 is used to pad the start of a song, so we need to add it
    unique_pitches = list(set([element["pitch"] for element in encoded_song] + [1]))
    unique_durations = list(set([element["duration"] for element in encoded_song] + [0]))
    unique_pitches.sort()
    unique_durations.sort()

    # Create dictionaries that map each name and value to an integer ID
    _pitch_to_id = {name: i for i, name in enumerate(unique_pitches)}
    _duration_to_id = {value: i for i, value in enumerate(unique_durations)}

    with open(os.path.join(MAPPING_PATH, "pitch_mapping.json"), "w") as _pitch_file:
        json.dump(_pitch_to_id, _pitch_file, indent=4)

    with open(os.path.join(MAPPING_PATH, "duration_mapping.json"), "w") as _duration_file:
        json.dump(_duration_to_id, _duration_file, indent=4)


def bulk_append(dict_list, new_dict):
    for key, val in new_dict.items():
        dict_list[key].append(val)


def process_input_data(data, tones=None, return_only_last_future_seq=False):
    pos = 0
    # treat as anticipation if long rest at start of song
    if data[0]["pitch"] == 0 and data[0]["duration"] >= 8:
        pos = -16

    onehot_vector_inputs = {k: [] for k in input_params}
    future_words_seq = []
    tone_data = data if tones is None else tones

    for index, element in enumerate(data):
        pitch = element["pitch"]
        duration = element["duration"]

        pitch_id = pitch_to_id[str(pitch)]
        duration_id = duration_to_id[str(duration)]

        pos += duration
        when_end = len(tone_data) - index - 1
        when_rest = when_end
        if not return_only_last_future_seq:
            future_words_onehot = []
            for idx in reversed(range(index, index + SEQUENCE_LENGTH)):
                tone = END_TONE if idx >= len(tone_data) else tone_data[idx]["tone"]
                phrasing = num_phrasing - 1 if idx >= len(tone_data) else tone_data[idx]["phrasing"]
                future_words_onehot.append([int(tone == j) for j in range(num_tone)] + [int(phrasing == j) for j in range(num_phrasing)])

            future_words_seq.append(future_words_onehot)

        for idx in range(index, len(tone_data)):
            if tone_data[idx]["tone"] in {REST_TONE, LONG_REST_TONE, PAD_TONE, END_TONE}:
                when_rest = idx - index

        current_tone = tone_data[index]["tone"]
        next_tone = tone_data[index + 1]["tone"] if index + 1 < len(tone_data) else END_TONE
        next_pitch = data[index + 1]["pitch"] if index + 1 < len(data) else 0

        single_input = {
            "pitch": pitch_id,
            "duration": duration_id,
            "pos_internal": pos % 16,
            "pos_external": (pos // 16) % 4,
            "current_tone": current_tone,
            "next_tone": next_tone,
            "when_end": min(when_end, num_when_end - 1),
            "when_rest": min(when_rest, num_when_rest - 1),
            "phrasing": element["phrasing"] if tones is None else tones[index]["phrasing"]
        }

        single_input = {k: [int(v == j) for j in range(param_shapes[k])] for k, v in single_input.items()}
        single_input["valid_pitches"] = calculate_valid_pitches(current_tone, next_tone,
                                                                pitch, next_pitch)

        bulk_append(onehot_vector_inputs, single_input)
    
    if return_only_last_future_seq:
        for idx in reversed(range(len(data) - 1, len(data) + SEQUENCE_LENGTH)):
            tone = END_TONE if idx >= len(tone_data) else tone_data[idx]["tone"]
            phrasing = num_phrasing - 1 if idx >= len(tone_data) else tone_data[idx]["phrasing"]
            future_words_seq.append([int(tone == j) for j in range(num_tone)] + [int(phrasing == j) for j in range(num_phrasing)])

    return onehot_vector_inputs, future_words_seq


def calculate_valid_pitches(current_tone, next_tone, current_pitch, next_pitch):
    interval_idx = f"{current_tone}_{next_tone}"
    valid_pitches = [0] * num_pitch

    if next_tone in (REST_TONE, LONG_REST_TONE, END_TONE, PAD_TONE, SEPARATOR_TONE):
        valid_pitches[pitch_to_id["0"]] = 1

    elif interval_idx not in tones_to_intervals:
        valid_pitches = [1] * num_pitch
        valid_pitches[pitch_to_id["0"]] = valid_pitches[pitch_to_id["1"]] = 0

    else:
        for interval in tones_to_intervals[interval_idx]:
            new_pitch = current_pitch + interval
            if str(new_pitch) in pitch_to_id:
                valid_pitches[pitch_to_id[str(new_pitch)]] = 1

    return valid_pitches


def process_output_data(data):
    onehot_vector_outputs = {k: [] for k in output_params}

    for element in data:
        pitch = int(element["pitch"])
        duration = int(element["duration"])

        pitch_id = pitch_to_id[str(pitch)]
        duration_id = duration_to_id[str(duration)]

        # Create a list of one-hot encoded vectors for each element
        bulk_append(onehot_vector_outputs, {
            "pitch": [int(pitch_id == i) for i in range(num_pitch)],
            "duration": [int(duration_id == j) for j in range(num_duration)],
        })

    return onehot_vector_outputs


def generating_training_sequences(dataset_path=DATASET_PATH, randomise_dataset=False):
    # Give the network 4 bars of notes (64 time steps) and 4 bar of tones, with the tone that the target has

    inputs = {k: [] for k in input_params}
    outputs = {k: [] for k in output_params}

    for path, _, files in os.walk(dataset_path):
        for fileId, filename in enumerate(files):
            if filename == '.DS_Store':
                continue

            stdout.write(f"\rProcessing {filename}   ({fileId + 1} / {len(files)})")
            stdout.flush()
            for _ in range(round((random.random() ** 0.5) + (random.random() ** 2) * 2) if randomise_dataset else 1):
                with open(os.path.join(path, filename)) as file:
                    data = json.load(file)
                    data = [PAD_NOTE] * SEQUENCE_LENGTH + data

                    no_of_inputs = len(data) - SEQUENCE_LENGTH

                    onehot_vector_inputs, future_words_seq = process_input_data(data)
                    onehot_vector_outputs = process_output_data(data)
                    for i in range(no_of_inputs):
                        temp_inputs = {k: v[i:(i + SEQUENCE_LENGTH)] for k, v in onehot_vector_inputs.items()}
                        temp_inputs["future_words"] = future_words_seq[i]
                        bulk_append(inputs, temp_inputs)
                        bulk_append(outputs, {k: v[i + SEQUENCE_LENGTH] for k, v in onehot_vector_outputs.items()})
                    # Try forcing the model to handle start and end better
                    for _ in range(3):
                        for i in range(max(0, no_of_inputs - 20), no_of_inputs): 
                            temp_inputs = {k: v[i:(i + SEQUENCE_LENGTH)] for k, v in onehot_vector_inputs.items()}
                            temp_inputs["future_words"] = future_words_seq[i]
                            bulk_append(inputs, temp_inputs)
                            bulk_append(outputs, {k: v[i + SEQUENCE_LENGTH] for k, v in onehot_vector_outputs.items()})
                        for i in range(0, min(no_of_inputs, 20)): 
                            temp_inputs = {k: v[i:(i + SEQUENCE_LENGTH)] for k, v in onehot_vector_inputs.items()}
                            temp_inputs["future_words"] = future_words_seq[i]
                            bulk_append(inputs, temp_inputs)
                            bulk_append(outputs, {k: v[i + SEQUENCE_LENGTH] for k, v in onehot_vector_outputs.items()})

                

    print("\nFinished file processing.")

    print(f"There are {len(inputs['pitch'])} sequences.")

    inputs = {k: np.array(v) for k, v in inputs.items()}
    outputs = {k: np.array(v) for k, v in outputs.items()}

    return inputs, outputs


def main():
    create_datasets_and_mapping([RAW_DATA_PATH, RAW_TESTING_DATA_PATH], [DATASET_PATH, TESTING_DATASET_PATH])


if __name__ == "__main__":
    main()
