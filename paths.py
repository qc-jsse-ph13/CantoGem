import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

BUILD_PATH = "build"
SAVE_MODEL_PATH = os.path.join(BUILD_PATH, "model_weights.ckpt")
PLOT_PATH = os.path.join(BUILD_PATH, "training_plot.png")

SERVER_DATABASE_PATH = "server_database"

MIDI_SAVE_PATH = "melody.mid"
MELODY_MIDI_SAVE_PATH = "melody_voice.mid"
MELODY_HIGH_MIDI_SAVE_PATH = "melody_voice_strings.mid"
CHORD_MIDI_SAVE_PATH = "melody_chords.mid"
BASS_MIDI_SAVE_PATH = "melody_bass.mid"
ALBERTI_BASS_MIDI_SAVE_PATH = "alberti_bass.mid"
PROGRESS_PATH = "progressbar.txt"

MELODY_WAV_PATH = "melody.wav"
MELODY_2_WAV_PATH = "melody2.wav"
CHORD_WAV_PATH = "chords.wav"
BASS_WAV_PATH = "bass.wav"

MP3_PATH = "song.mp3"
SOUNDFONT_MELODY_PATH = os.path.join(CURRENT_DIR, "soundfonts", "e_guitar.sf2")
SOUNDFONT_HIGH_MELODY_PATH = os.path.join(CURRENT_DIR, "soundfonts", "synth_strings.sf2")
SOUNDFONT_BASS_PATH = os.path.join(CURRENT_DIR, "soundfonts", "piano.sf2")
SOUNDFONT_ALBERTI_BASS_PATH = os.path.join(CURRENT_DIR, "soundfonts", "80s_okmoog_bass.sf2")
SOUNDFONT_CHORD_PATH = os.path.join(CURRENT_DIR, "soundfonts", "c_guitar.sf2")


def get_user_data_path(userid, filepath=""):
    if filepath == "":
        return os.path.join(SERVER_DATABASE_PATH, userid)
    return os.path.join(SERVER_DATABASE_PATH, userid, filepath)
