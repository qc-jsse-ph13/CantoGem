import numpy as np

CHORD_DURATION = 8

'''
IMPORTANT NOTE:
All notes in the harmonizer are wrapped into 0-12 which represents do-te, except for construct_chords()
'''


class Chord:
    def __init__(self, scale_degree, base_note, quality, name=""):
        self.scale_degree = scale_degree
        self.base_note = base_note
        self.quality = quality
        self.name = name

    def get_score(self):
        score = np.zeros(12)
        if self.quality == "major":
            # Favour diatonic notes
            score[self.base_note % 12] = score[(self.base_note + 4) % 12] = score[(self.base_note + 7) % 12] = 1
            # Slightly favour chordal seventh (M7) but don't it form a dominant seventh (m7)
            score[(self.base_note + 11) % 12] = 0.7
            # Favour second, fourth and sixth suspension the least
            score[(self.base_note + 2) % 12] = score[(self.base_note + 9) % 12] = 0.3
            score[(self.base_note + 5) % 12] = 0.1
            # Always choose Emaj if G# exists
            if self.scale_degree == 3:
                score.fill(-1)
                score[8] = 1000
        elif self.quality == "minor":
            score[self.base_note % 12] = score[(self.base_note + 3) % 12] = score[(self.base_note + 7) % 12] = 1
            score[(self.base_note + 10) % 12] = 0.7
            score[(self.base_note + 2) % 12] = score[(self.base_note + 5) % 12] = 0.3
            score[(self.base_note + 8) % 12] = 0.1
        elif self.quality == "diminished":
            score[self.base_note % 12] = score[(self.base_note + 3) % 12] = score[(self.base_note + 6) % 12] = 1
            score[(self.base_note + 10) % 12] = score[(self.base_note + 9) % 12] = 0.5
            score[(self.base_note + 5) % 12] = 0.1
            # Always choose C dominant7 if BFlat
            if self.scale_degree == 3:
                score.fill(-1)
                score[10] = 1000

        return score

    def construct_chord(self, ref_do):
        if self.quality == "major":
            return [ref_do + self.base_note + interval for interval in [0, 4, 7]]
        elif self.quality == "minor":
            return [ref_do + self.base_note + interval for interval in [0, 3, 7]]
        elif self.quality == "diminished":
            return [ref_do + self.base_note + interval for interval in [0, 3, 6]]
        return

    def __str__(self):
        return f"{self.scale_degree}{self.quality}"


chord_list = {
    "I": Chord(1, 0, "major", "Cmaj"),
    "II": Chord(2, 2, "minor", "Dmin"),
    "iii": Chord(3, 4, "minor", "Emin"),
    "III": Chord(3, 4, "major", "Emaj"),
    "bFlat": Chord(3, 4, "diminished", "Cdom7"),
    "IV": Chord(4, 5, "major", "Fmaj"),
    "V": Chord(5, 7 - 12, "major", "Gmaj"),
    "vi": Chord(6, 9 - 12, "minor", "Amin"),
    "vii": Chord(7, 11 - 12, "diminished", "Bdim"),
}


def min_steps(arr, a, b):
    n = len(arr)
    if a < b:
        return min(b - a, n - b + a)
    else:
        return min(a - b, n - a + b)


def get_score(melody_extract, chord, previous_chords, when_end):
    diatonic_tones_score = sum(chord.get_score()[note] for note in melody_extract)

    # Handle starting chords
    if len(previous_chords) <= 3:
        starting_score = 0
        if len(previous_chords) == 0:
            if chord.scale_degree == 1 or chord.scale_degree == 4:
                starting_score += 1
        return diatonic_tones_score + starting_score

    # Handle where there are no rests
    if len(melody_extract) == 0:
        return 100 if chord == previous_chords[-1] else 0

    musicality_score = 0

    # Handle repeating chords
    if previous_chords[-1] == chord:
        musicality_score += -1 if when_end > 2 else 1
        if previous_chords[-2] == previous_chords[-1]:
            musicality_score += -2 if when_end > 2 else 1

    # Favour descending fifths and descending thirds
    prev2_deg = previous_chords[-2].scale_degree
    prev_deg = previous_chords[-1].scale_degree
    cur_deg = chord.scale_degree

    prev_downward_difference = prev2_deg - prev_deg if prev2_deg > prev_deg else 7 - (prev_deg - prev2_deg)
    downward_difference = prev_deg - cur_deg if prev_deg > cur_deg else 7 - (cur_deg - prev_deg)

    # Descending thirds
    if prev_downward_difference == 4 and downward_difference == 6:
        musicality_score += 1
    # Descending fifths
    if downward_difference == 4:
        musicality_score += 1

    # Functional harmony (No need repeatedly handle descending fifths)
    if previous_chords[-1].scale_degree == 1:
        pass
    elif previous_chords[-1].name == "Emaj":
        if chord.name not in {"Emaj", "Amin"}:
            musicality_score -= 2
    elif previous_chords[-1].name == "Cdom7":
        if chord.name not in {"Cdom7", "Fmaj"}:
            musicality_score -= 2
    elif previous_chords[-1].scale_degree == 2:
        if when_end <= 3 and chord.scale_degree == 5:
            musicality_score += 1
        if chord.scale_degree == 1:
            musicality_score -= 1
        elif chord.scale_degree == 4 or chord.scale_degree == 6:
            musicality_score += 1

    elif previous_chords[-1].scale_degree == 3:
        if chord.scale_degree == 1:
            musicality_score -= 1

    elif previous_chords[-1].scale_degree == 4:
        if chord.scale_degree == 2:
            musicality_score += 0.5
        elif chord.scale_degree == 6:
            musicality_score += 0.5
        if chord.scale_degree == 5:
            musicality_score += 1

    elif previous_chords[-1].scale_degree == 5:
        if chord.scale_degree == 6:
            musicality_score += 0.5
        elif chord.scale_degree == 3:
            musicality_score += 0.5
        # Prevent it from reaching perfect cadence
        elif chord.scale_degree == 1 and when_end > 3:
            musicality_score -= 1

    elif previous_chords[-1].scale_degree == 6:
        if chord.scale_degree == 1:
            musicality_score -= 1
        elif chord.scale_degree == 5:
            musicality_score += 0.5

    elif previous_chords[-1].scale_degree == 7:
        if chord.scale_degree == 6:
            musicality_score += 0.5

    if when_end <= 3:
        if chord.scale_degree == 1:
            musicality_score += 0.5
        if chord.scale_degree == 5 or chord.scale_degree == 1 or chord.scale_degree == 6:
            musicality_score += 2

    return diatonic_tones_score + musicality_score


def harmonise(melody):
    """
    Format the melody:
    I: {pitch = 62, duration = 4}
    O: [62, 62, 62, 62]
    Rests are represented by -1, which are removed and not considered later on
    """
    formatted_melody = []
    target_chords = []
    for note in melody:
        for i in range(note["duration"]):
            if note["pitch"] == 0:
                formatted_melody.append(-1)
            else:
                rel_pitch = (note["pitch"] - 60)
                formatted_melody.append(rel_pitch % 12 if rel_pitch >= 0 else (12 + rel_pitch) % 12)

    # Consider one chordal duration at a time
    for i in range(0, len(formatted_melody), CHORD_DURATION):
        # Extract the selected melody and remove all rests
        melody_extract = formatted_melody[i:i + CHORD_DURATION]
        melody_extract = [x for x in melody_extract if x != -1]

        # Find the chord with the highest score (plausibility)
        chord = max(chord_list.values(), key=lambda _chord: get_score(melody_extract, _chord, target_chords,
                                                                      (len(formatted_melody) - i) // CHORD_DURATION))
        target_chords.append(chord)
    """
    if target_chords[-1].scale_degree == 3:
        target_chords.append(chord_list["vi"])
    if target_chords[-1].scale_degree == 2 or target_chords[-1].scale_degree == 4:
        target_chords.append(chord_list["V"])
    if target_chords[-1].scale_degree == 5:
        target_chords.append(chord_list["I"])
    """

    return target_chords
