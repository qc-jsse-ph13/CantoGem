import os
from pydub import AudioSegment

from midi2audio import FluidSynth
from paths import SOUNDFONT_MELODY_PATH, SOUNDFONT_CHORD_PATH, MELODY_MIDI_SAVE_PATH, CHORD_MIDI_SAVE_PATH, BASS_WAV_PATH, \
    MELODY_WAV_PATH, CHORD_WAV_PATH, MP3_PATH, SOUNDFONT_BASS_PATH, SOUNDFONT_ALBERTI_BASS_PATH, BASS_MIDI_SAVE_PATH,\
    ALBERTI_BASS_MIDI_SAVE_PATH, MELODY_2_WAV_PATH, MELODY_HIGH_MIDI_SAVE_PATH, SOUNDFONT_HIGH_MELODY_PATH
from paths import get_user_data_path

fs1 = FluidSynth(SOUNDFONT_MELODY_PATH)
fs2 = FluidSynth(SOUNDFONT_CHORD_PATH)
fs3 = FluidSynth(SOUNDFONT_BASS_PATH)
fs4 = FluidSynth(SOUNDFONT_ALBERTI_BASS_PATH)
fs5 = FluidSynth(SOUNDFONT_HIGH_MELODY_PATH)


def synthesize(userid="testing_user_id"):
    melody_midi_save_path = get_user_data_path(userid, MELODY_MIDI_SAVE_PATH)
    melody_high_midi_save_path = get_user_data_path(userid, MELODY_HIGH_MIDI_SAVE_PATH)
    chord_midi_save_path = get_user_data_path(userid, CHORD_MIDI_SAVE_PATH)
    bass_midi_save_path = get_user_data_path(userid, BASS_MIDI_SAVE_PATH)
    alberti_bass_midi_save_path = get_user_data_path(userid, ALBERTI_BASS_MIDI_SAVE_PATH)

    melody_wav_path = get_user_data_path(userid, MELODY_WAV_PATH)
    melody_2_wav_path = get_user_data_path(userid, MELODY_2_WAV_PATH)
    chord_wav_path = get_user_data_path(userid, CHORD_WAV_PATH)
    bass_wav_path = get_user_data_path(userid, BASS_WAV_PATH)
    mp3_path = get_user_data_path(userid, MP3_PATH)

    fs1.midi_to_audio(melody_midi_save_path, melody_wav_path)
    fs2.midi_to_audio(chord_midi_save_path, chord_wav_path)
    fs3.midi_to_audio(bass_midi_save_path, bass_wav_path)
    fs5.midi_to_audio(melody_high_midi_save_path, melody_2_wav_path)

    melody = AudioSegment.from_file(melody_wav_path, format="wav") + 7
    melody_2 = AudioSegment.from_file(melody_2_wav_path, format="wav")
    chords = AudioSegment.from_file(chord_wav_path, format="wav")
    bass = AudioSegment.from_file(bass_wav_path, format="wav")

    melody\
        .overlay(melody_2)\
        .overlay(chords)\
        .overlay(bass)\
        .export(melody_wav_path, format="wav")

    fs4.midi_to_audio(alberti_bass_midi_save_path, bass_wav_path)

    melody = AudioSegment.from_file(melody_wav_path, format="wav")
    bass = AudioSegment.from_file(bass_wav_path, format="wav")

    if os.path.exists(mp3_path):
        os.remove(mp3_path)

    melody\
        .overlay(bass)\
        .export(mp3_path)

    os.remove(melody_wav_path)
    os.remove(melody_2_wav_path)
    os.remove(chord_wav_path)
    os.remove(bass_wav_path)


if __name__ == "__main__":
    synthesize()
