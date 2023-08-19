import math
import os
import random
import re
import json
from enum import Enum

import numpy as np
import music21 as m21
import pycantonese as pc
from pycantonese.word_segmentation import Segmenter

from preprocess import id_to_pitch, id_to_duration, duration_to_id, pitch_to_id
from preprocess import input_params, output_params
from preprocess import process_input_data

from preprocess import SEQUENCE_LENGTH, PAD_TONE, REST_TONE, LONG_REST_TONE

from train import build_model

from harmoniser import harmonise, CHORD_DURATION
from synthesizer import synthesize

from paths import SAVE_MODEL_PATH, MELODY_MIDI_SAVE_PATH, CHORD_MIDI_SAVE_PATH, MIDI_SAVE_PATH, PROGRESS_PATH, \
    get_user_data_path, BASS_MIDI_SAVE_PATH, BUILD_PATH, ALBERTI_BASS_MIDI_SAVE_PATH, MELODY_HIGH_MIDI_SAVE_PATH

from flask import abort

CHORD_REFERENCE_DO = 48

RANGE = 19  # 1.5 octaves


class Voice(Enum):  # Lower limit of voices
    BASS = 53
    TENOR = 60
    ALTO = 55
    SOPRANO = 57


def _sample_with_temperature(probabilities, temperature):
    # temperature -> infinity -> Homogenous distribution
    # temperature -> 0 -> deterministic
    # temperature -> 1 -> keep probabilities
    # normalised probabilities again to guarantee floating point errors won't round stuff down to 0 in power step

    probabilities = probabilities / np.sum(probabilities)
    probabilities = np.power(probabilities, 1 / temperature)
    probabilities = probabilities / np.sum(probabilities)

    choices = range(len(probabilities))
    index = np.random.choice(choices, p=probabilities)

    return index


def onehot_input_from_seed(data, tones):
    onehot_input_dict, future_words_seq = process_input_data(
        [{"pitch": 1, "duration": 0} for _ in range(SEQUENCE_LENGTH)] + data,
        [{"tone": PAD_TONE, "phrasing": 0} for _ in range(SEQUENCE_LENGTH)] + tones,
        return_only_last_future_seq=True
    )
    onehot_input_dict["future_words"] = future_words_seq
    onehot_input = [onehot_input_dict[k][-SEQUENCE_LENGTH:] for k in input_params]

    for i, onehot_vectors in enumerate(onehot_input):
        onehot_input[i] = np.array(onehot_vectors)[np.newaxis, ...]
    return onehot_input


def save_song(melody, voice, userid):
    transposition_factor = get_transposition_factor(melody, voice)
    chord_progression = harmonise(melody)
    save_melody(melody, transposition_factor, userid)
    save_chords(chord_progression, transposition_factor, userid)
    save_bass(chord_progression, transposition_factor, userid)

    stream = m21.stream.Stream()
    stream.insert(0, m21.converter.parse(get_user_data_path(userid, MELODY_MIDI_SAVE_PATH)).flatten())
    stream.insert(0, m21.converter.parse(get_user_data_path(userid, MELODY_HIGH_MIDI_SAVE_PATH)).flatten())
    stream.insert(0, m21.converter.parse(get_user_data_path(userid, CHORD_MIDI_SAVE_PATH)).flatten())
    stream.insert(0, m21.converter.parse(get_user_data_path(userid, BASS_MIDI_SAVE_PATH)).flatten())
    stream.insert(0, m21.converter.parse(get_user_data_path(userid, ALBERTI_BASS_MIDI_SAVE_PATH)).flatten())
    stream.write("midi", get_user_data_path(userid, MIDI_SAVE_PATH))

    synthesize(userid)


def save_melody(melody, transposition_factor, userid):
    stream = m21.stream.Stream()
    lengthened_melody = []
    for note in melody:
        if note["duration"] == 0:
            continue
        if note["pitch"] < 10:
            m21_event = m21.note.Rest(quarterLength=note["duration"] / 4)
            if not len(lengthened_melody):
                lengthened_melody.append(note.copy())
            else:
                lengthened_melody[-1]["duration"] += note["duration"]
        else:
            m21_event = m21.note.Note(note["pitch"], quarterLength=note["duration"] / 4)
            if not len(lengthened_melody) or ((note["duration"] >= 4 or lengthened_melody[-1]["pitch"] < 10) and\
                note["pitch"] != lengthened_melody[-1]["pitch"]):
                lengthened_melody.append(note.copy())
            else:
                lengthened_melody[-1]["duration"] += note["duration"]
        stream.append(m21_event)
    stream.transpose(transposition_factor).write("midi", get_user_data_path(userid, MELODY_MIDI_SAVE_PATH))
    stream = m21.stream.Stream()
    for note in lengthened_melody:
        if note["pitch"] < 10:
            m21_event = m21.note.Rest(quarterLength=note["duration"] / 4)
        else:
            m21_event = m21.note.Note(note["pitch"], quarterLength=note["duration"] / 4)
        stream.append(m21_event)
    stream.transpose(transposition_factor + 12).write("midi", get_user_data_path(userid, MELODY_HIGH_MIDI_SAVE_PATH))


def save_bass(chord_progression, transposition_factor, userid):
    chords = [chord.construct_chord(CHORD_REFERENCE_DO) for chord in chord_progression]

    new_stream = m21.stream.Stream()
    alberti_stream = m21.stream.Stream()
    for i in range(0, len(chords) * CHORD_DURATION, CHORD_DURATION):
        new_chord = chords[i // CHORD_DURATION]
        new_chord_object = m21.chord.Chord([new_chord[0] - 12, new_chord[0] - 24],
                                           quarterLength=CHORD_DURATION / 4 * 2)
        new_chord_object.volume = m21.volume.Volume(velocity=75)
        new_stream.insert(i / 4, new_chord_object)

        base_note = m21.note.Note(new_chord[0] - 12, quarterLength=CHORD_DURATION / 4)
        mid_note = m21.note.Note(new_chord[2] - 12, quarterLength=CHORD_DURATION / 16)
        high_note = m21.note.Note(new_chord[1], quarterLength=CHORD_DURATION / 16)
        base_note.volume = m21.volume.Volume(velocity=90)
        mid_note.volume = high_note.volume = m21.volume.Volume(velocity=75)
        alberti_stream.insert(i / 4, base_note)
        if i != (len(chords) - 1) * CHORD_DURATION:
            alberti_stream.insert((i + 1 * CHORD_DURATION / 4) / 4, mid_note)
            alberti_stream.insert((i + 2 * CHORD_DURATION / 4) / 4, high_note)
            alberti_stream.insert((i + 3 * CHORD_DURATION / 4) / 4, mid_note.__deepcopy__())

    new_stream.transpose(transposition_factor).write("midi", get_user_data_path(userid, BASS_MIDI_SAVE_PATH))
    alberti_stream.transpose(transposition_factor).write("midi", get_user_data_path(userid, ALBERTI_BASS_MIDI_SAVE_PATH))


def save_chords(chord_progression, transposition_factor, userid):

    chords = [chord.construct_chord(CHORD_REFERENCE_DO) for chord in chord_progression]
    strum_timesteps = [0, 1, 1.5, 2.5, 3, 3.5]

    new_stream = m21.stream.Stream()
    for i in range(0, len(chords) * CHORD_DURATION, CHORD_DURATION):
        new_chord = chords[i // CHORD_DURATION]
        if i == (len(chords) - 1) * CHORD_DURATION:
            new_chord_object = m21.chord.Chord([new_chord[0] - 12, new_chord[2] - 12, new_chord[0]],
                                               quarterLength=CHORD_DURATION / 4 * 2)
            new_chord_object.volume = m21.volume.Volume(velocity=75)
            new_stream.insert(i / 4, new_chord_object)
        else:
            for j, timestep in enumerate(strum_timesteps):
                low_note = m21.note.Note(new_chord[2] - 12, quarterLength=CHORD_DURATION / 4)
                mid_note = m21.note.Note(new_chord[0], quarterLength=CHORD_DURATION / 4)
                high_note = m21.note.Note(new_chord[1], quarterLength=CHORD_DURATION / 4)

                low_note.volume = mid_note.volume = high_note.volume = m21.volume.Volume(velocity=50 if j % 2 == 0 else 35)
                new_stream.insert((i + timestep * CHORD_DURATION / 4) / 4, low_note)
                new_stream.insert((i + timestep * CHORD_DURATION / 4) / 4, mid_note)
                new_stream.insert((i + timestep * CHORD_DURATION / 4) / 4, high_note)

            low_note = m21.note.Note(new_chord[2] - 12, quarterLength=CHORD_DURATION / 4)
            mid_note = m21.note.Note(new_chord[0], quarterLength=CHORD_DURATION / 4)
            high_note = m21.note.Note(new_chord[1], quarterLength=CHORD_DURATION / 4)

            low_note.volume = mid_note.volume = high_note.volume = m21.volume.Volume(velocity=50)

            new_stream.insert((i + 0 * CHORD_DURATION / 4) / 4, low_note)
            new_stream.insert((i + 1 * CHORD_DURATION / 4) / 4, mid_note)
            new_stream.insert((i + 2 * CHORD_DURATION / 4) / 4, high_note)
            new_stream.insert((i + 3 * CHORD_DURATION / 4) / 4, mid_note.__deepcopy__())

    new_stream.transpose(transposition_factor).write("midi", get_user_data_path(userid, CHORD_MIDI_SAVE_PATH))


def get_transposition_factor(melody, voice):
    lowest_note = 1000
    for event in melody:
        # Reserve bottom 10 notes for other functions
        if event["pitch"] > 10:
            lowest_note = min(lowest_note, event["pitch"])

    return voice.value - lowest_note + random.choice([0, 1, -1])

class MelodyGenerator:

    def __init__(self, model_path=SAVE_MODEL_PATH):
        self.model_path = model_path
        self.model = build_model()
        self.model.load_weights(model_path).expect_partial()

    def generate_melody(self, all_tones, userid):
        # create seed with start symbols
        current_melody = []
        if not os.path.isdir(get_user_data_path(userid)):
            os.makedirs(get_user_data_path(userid))

        with open(get_user_data_path(userid, PROGRESS_PATH), "w") as progressbar_file:
            progressbar_file.write("0.00%")

        for i in range(len(all_tones)):
            # create seed with start symbols
            onehot_seed = onehot_input_from_seed(current_melody, all_tones)

            valid_pitches = onehot_seed[input_params.index("valid_pitches")][0, -1]

            # make a prediction
            probabilities = self.model.predict(onehot_seed)
            probabilities[output_params.index("duration")][0][duration_to_id["0"]] = 0.0
            # if np.sum(probabilities[output_params.index("pitch")][0] * (valid_pitches + 0.000005)) > 0:
            #    probabilities[output_params.index("pitch")][0] *= (valid_pitches + 0.000005)
            probabilities[output_params.index("pitch")][0][pitch_to_id["1"]] = 0.0
            if all_tones[i]["tone"] in {LONG_REST_TONE, REST_TONE, PAD_TONE}:
                probabilities[output_params.index("pitch")][0][pitch_to_id["0"]] = 10000.0
            else:
                probabilities[output_params.index("pitch")][0][pitch_to_id["0"]] = 0.0

            # choose semi-random note from probability distribution (pitch class, duration class)
            # temperature: temperature[key]((_ + 1) / len(all_tones)
            output_note = {
                key: _sample_with_temperature(probabilities[index][0], 0.1)
                for index, key in enumerate(output_params)
            }

            output_note["pitch"] = id_to_pitch[output_note["pitch"]]
            output_note["duration"] = id_to_duration[output_note["duration"]]
            print(output_note)

            current_melody.append(output_note)

            with open(get_user_data_path(userid, PROGRESS_PATH), "w") as progressbar_file:
                progressbar_file.write("{:.2f}%".format((i + 1) * 100 / len(all_tones)))

        print(current_melody)

        return current_melody


def get_bell_sigmoid(min_val, max_val, roughness):
    k = (max_val - min_val) * (1 + math.exp(-roughness / 6)) / (1 - math.exp(-roughness / 6))
    return lambda x: k / (1 + math.exp(roughness * (1 / 3 - x))) + k / (
            1 + math.exp(roughness * (x - 2 / 3))) - k + min_val


def parse_lyrics(lyrics):
    try:
        print("parsing lyrics:", lyrics)
        rest_tones_pos = []
        for char in lyrics:
            if char == ",":
                rest_tones_pos.append(REST_TONE)
            elif char == "|":
                rest_tones_pos.append(LONG_REST_TONE)
            else:
                rest_tones_pos.append(0)

        pure_words = lyrics.replace(",", "").replace("|", "")
        all_tones = []

        segmenter = Segmenter(disallow={"一個人"}, allow={"大江"})
        tokens = pc.parse_text(pure_words, segment_kwargs={"cls": segmenter}).tokens()

        for token in tokens:
            if token.word == "浪":
                token.jyutping = "long6"
            anglicised_words = re.split(r'(?<=[0-9])(?=[a-zA-Z])', token.jyutping)
            tones = [{"tone": int(char[-1]), "phrasing": len(anglicised_words) - idx} for idx, char in
                    enumerate(anglicised_words)]
            all_tones.extend(tones)

        for i in range(len(lyrics)):
            if rest_tones_pos[i] != 0:
                all_tones.insert(i, {"tone": rest_tones_pos[i], "phrasing": 5})

        print("Finished lyrics parsing")
        return all_tones
    except:
        print("You suck")
        abort(400)


mg_list = dict()
for raw_file_name in os.listdir(BUILD_PATH):
    file_name = os.path.join(BUILD_PATH, raw_file_name)
    if os.path.isfile(file_name) and raw_file_name[-11:] == ".ckpt.index":
        print(raw_file_name[:-11])
        mg_list[raw_file_name[:-11]] = MelodyGenerator(file_name[:-6])


def compress_melody_object(melody):
    return "|".join([str(note["pitch"]) + "," + str(note["duration"]) for note in melody])


def make_melody_response(lyrics, voice, userid="testing_user_id", filename="model_weights"):
    print("Starting melody generation")

    tones = parse_lyrics(lyrics)
    print("Generating")

    melody = mg_list[filename].generate_melody(tones, userid)
    save_song(melody, voice, userid)

    tpf = get_transposition_factor(melody, voice)
    for note in melody:
        if note["pitch"] > 10:
            note["pitch"] += tpf

    with open(get_user_data_path(userid, PROGRESS_PATH), "w") as progressbar_file:
        progressbar_file.write("0.00%")

    return compress_melody_object(melody)

def _change_stream_tempo(stream: m21.stream.Stream, new_tempo):
    new_stream = m21.stream.Stream()
    new_stream.append(m21.tempo.MetronomeMark(".", new_tempo, m21.note.Note(type="quarter")))
    new_stream.append(stream.flatten().notesAndRests.stream())
    return new_stream

def change_tempo(userid, new_tempo):
    melody_stream = m21.converter.parse(get_user_data_path(userid, MELODY_MIDI_SAVE_PATH))
    melody_high_stream = m21.converter.parse(get_user_data_path(userid, MELODY_HIGH_MIDI_SAVE_PATH))
    chord_stream = m21.converter.parse(get_user_data_path(userid, CHORD_MIDI_SAVE_PATH))
    bass_stream = m21.converter.parse(get_user_data_path(userid, BASS_MIDI_SAVE_PATH))
    alberti_bass_stream = m21.converter.parse(get_user_data_path(userid, ALBERTI_BASS_MIDI_SAVE_PATH))
    melody_stream = _change_stream_tempo(melody_stream, new_tempo)
    melody_high_stream = _change_stream_tempo(melody_high_stream, new_tempo)
    chord_stream = _change_stream_tempo(chord_stream, new_tempo)
    bass_stream = _change_stream_tempo(bass_stream, new_tempo)
    alberti_bass_stream = _change_stream_tempo(alberti_bass_stream, new_tempo)
    chord_stream.show("text")
    melody_stream.write("midi", get_user_data_path(userid, MELODY_MIDI_SAVE_PATH))
    melody_high_stream.write("midi", get_user_data_path(userid, MELODY_HIGH_MIDI_SAVE_PATH))
    chord_stream.write("midi", get_user_data_path(userid, CHORD_MIDI_SAVE_PATH))
    bass_stream.write("midi", get_user_data_path(userid, BASS_MIDI_SAVE_PATH))
    alberti_bass_stream.write("midi", get_user_data_path(userid, ALBERTI_BASS_MIDI_SAVE_PATH))
    synthesize(userid)


if __name__ == "__main__":
    make_melody_response(",大江東去,浪淘盡,千古風流人物|"
                         "故壘西邊,人道是,三國周郎赤壁|"
                         "亂石崩雲,驚濤裂岸,捲起千堆雪|"
                         "江山如畫,一時多少豪傑|"
                         "遙想公瑾當年,小喬初嫁了,雄姿英發|"
                         "羽扇綸巾,談笑間,檣櫓灰飛煙滅|"
                         "故國神遊,多情應笑我,早生華髮|"
                         "人生如夢,一尊還酹江月", Voice.TENOR)

