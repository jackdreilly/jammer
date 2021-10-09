import itertools
from pathlib import Path

import pytest

from jammer import (
    Accidental,
    Chord,
    ChordProgression,
    Letter,
    NoteName,
    Pitch,
    major,
    make_midi,
)


def test_note_name_iterator():
    assert list(NoteName.iterator()) == list(
        map(
            NoteName.from_string,
            (
                "A",
                "A#",
                "B",
                "C",
                "C#",
                "D",
                "D#",
                "E",
                "F",
                "F#",
                "G",
                "G#",
            ),
        )
    )


def test_note_name_equal():
    assert NoteName.from_string("C#") == NoteName.from_string("Db")


def test_major():
    assert major.intervals == [0, 2, 4, 5, 7, 9, 11]


def test_triad():
    assert major.triad(1) == Chord(0, [4, 7])
    assert major.triad(2) == Chord(2, [5, 9])
    assert major.triad(5) == Chord(7, [11, 2])


def test_letter():
    assert Letter.from_string("a") == Letter.A
    assert Letter.from_string("A") == Letter.A
    assert Letter.from_string("c") == Letter.C


@pytest.mark.parametrize(
    ("string", "accidental"),
    (
        ("#", Accidental.sharp),
        ("b", Accidental.flat),
        ("n", Accidental.natural),
        ("♯", Accidental.sharp),
        ("♭", Accidental.flat),
        ("♮", Accidental.natural),
        ("", Accidental.natural),
    ),
)
def test_accidental(string: str, accidental: Accidental):
    assert Accidental.from_string(string) == accidental


@pytest.mark.parametrize(
    ("string", "note_name"),
    (
        ("A", NoteName(Letter.A, Accidental.natural)),
        ("a", NoteName(Letter.A, Accidental.natural)),
        ("B", NoteName(Letter.B, Accidental.natural)),
        ("bb", NoteName(Letter.B, Accidental.flat)),
        ("Bb", NoteName(Letter.B, Accidental.flat)),
        ("D#", NoteName(Letter.D, Accidental.sharp)),
        ("Dn", NoteName(Letter.D, Accidental.natural)),
    ),
)
def test_note_name_from_string(string: str, note_name: NoteName):
    assert NoteName.from_string(string) == note_name


@pytest.mark.parametrize(
    ("string", "pitch_midi_interval"),
    (("A4", 57), ("C4", 60), ("C", 60), ("C#4", 61), ("C#", 61), ("Cb", 59)),
)
def test_pitch_from_string(string: str, pitch_midi_interval: int):
    assert Pitch.from_string(string) == Pitch(pitch_midi_interval)


@pytest.mark.parametrize(
    ("string", "index"), (("A", 0), ("A#", 1), ("Bb", 1), ("C", 3))
)
def test_note_name_index(string: str, index: int):
    assert NoteName.from_string(string).index == index


def test_triad_pitched():
    assert [
        x.midi_interval for x in major.triad(1).pitched(Pitch.from_string("C")).pitches
    ] == [60, 64, 67]


@pytest.mark.parametrize(
    ("note_name", "octave"), itertools.product(NoteName.iterator(), range(5))
)
def test_pitch_note_name_octave(note_name: NoteName, octave: int):
    pitch = Pitch.from_note(note_name, octave)
    assert pitch.note_name == note_name
    assert pitch.octave == octave


@pytest.mark.parametrize("offset", range(12))
def test_note_name_a_offset(offset: int):
    assert NoteName.from_a_offset(offset) == list(NoteName.iterator())[offset]


def test_midi():
    make_midi(
        ChordProgression([1, 6, 2, 5]),
        Path(__file__).parent / "test.midi",
    )


@pytest.mark.parametrize("string", ("aasdf", "X", "XX", "CC", "#"))
def test_parse_fail_note_name(string):
    with pytest.raises(ValueError):
        NoteName.from_string(string)


@pytest.mark.parametrize("string", ("aasdf", "X4", "XX", "CC4", "#"))
def test_parse_fail_pitch(string):
    with pytest.raises(ValueError):
        Pitch.from_string(string)


def test_poop():
    assert False
