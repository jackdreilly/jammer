"""Test cases"""
import io
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from jet_jammer import (
    Accidental,
    BuilderValue,
    BuilderValueSequence,
    Chord,
    ChordProgression,
    Letter,
    Measure,
    MidiChord,
    MidiTrack,
    MidiTrackPlayBuilder,
    NoteName,
    Pitch,
    app,
    major,
    make_midi,
)


@dataclass(frozen=True)
class DictHas:
    kwargs: Dict[str, Any]

    def __eq__(self, other: dict) -> bool:
        return self.kwargs == {k: v for k, v in other.items() if k in self.kwargs}


def dict_has(**kwargs):
    return DictHas(kwargs=kwargs)


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


@pytest.mark.golden_test("goldens/*.yml")
def test_golden(golden):
    with io.BytesIO() as file_object:
        make_midi(
            chord_progression=ChordProgression([*range(1, 9), *range(8, 0, -1)]),
            file_object=file_object,
        )
        assert file_object.getvalue() == golden.out["output"]


def test_midi():
    with open(Path(__file__).parent / "test.midi", "wb") as fn:
        make_midi(
            chord_progression=ChordProgression([*range(1, 9), *range(8, 0, -1)]),
            file_object=fn,
        )


@pytest.mark.parametrize("string", ("aasdf", "X", "XX", "CC", "#"))
def test_parse_fail_note_name(string):
    with pytest.raises(ValueError):
        NoteName.from_string(string)


@pytest.mark.parametrize("string", ("aasdf", "X4", "XX", "CC4", "#", "aspdf9uh2"))
def test_parse_fail_pitch(string):
    with pytest.raises(ValueError):
        Pitch.from_string(string)


def test_fast_api():
    with io.BytesIO() as file_object:
        make_midi(
            chord_progression=ChordProgression([1, 3], Pitch.from_string("D#")),
            file_object=file_object,
            tempo=195,
        )
        in_memory_content = file_object.getvalue()
        assert in_memory_content
    response = TestClient(app).get(
        "/",
        params=[
            ["chord_numbers", 1],
            ["chord_numbers", 3],
            ["key", "D#"],
            ["tempo", 195],
        ],
    )
    assert response.headers == dict_has(
        **{
            "content-type": "audio/midi",
            "content-disposition": 'attachment; filename="jammer.midi"',
        }
    )
    assert response.content == in_memory_content


def test_sequence():
    assert MidiChord(1, [Pitch.from_string("C")]) * 2 == BuilderValueSequence(
        [
            BuilderValue.make(MidiChord(1, [Pitch.from_string("C")])),
            BuilderValue.make(MidiChord(1, [Pitch.from_string("C")])),
        ]
    )


def test_builder_measure():
    builder = MidiTrackPlayBuilder(MidiTrack("a"))
    assert builder.measure == 0
    builder.time = 4
    assert builder.measure == 1
    builder.time = 5
    assert builder.measure == 1
    builder.measure += 1
    assert builder.measure == 2
    assert builder.time == 8


def test_measure_mult():
    assert Measure(1) * 2 == Measure(2)
    assert 2 * Measure(3) == Measure(6)
