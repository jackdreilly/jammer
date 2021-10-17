"""Test cases"""
import io
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Set

import pytest
from fastapi.testclient import TestClient

from jet_jammer import (
    Accidental,
    BassNoteName,
    BuilderValue,
    BuilderValueSequence,
    Chord,
    ChordName,
    ChordNameProgression,
    ChordProgression,
    ChordType,
    Letter,
    Measure,
    MidiChord,
    MidiTrack,
    MidiTrackPlayBuilder,
    NoteName,
    Octave,
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
            NoteName.parse,
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
    assert NoteName.parse("C#") == NoteName.parse("Db")


def test_major():
    assert major.intervals == [0, 2, 4, 5, 7, 9, 11]


def test_triad():
    assert major.triad(1) == Chord(0, [4, 7])
    assert major.triad(2) == Chord(2, [5, 9])
    assert major.triad(5) == Chord(7, [11, 2])


def test_letter():
    assert Letter.parse("a") == Letter.A
    assert Letter.parse("A") == Letter.A
    assert Letter.parse("c") == Letter.C


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
    assert Accidental.parse(string) == accidental


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
def test_note_name_parse(string: str, note_name: NoteName):
    assert NoteName.parse(string) == note_name


@pytest.mark.parametrize(
    ("string", "pitch_midi_interval"),
    (("A4", 57), ("C4", 60), ("C", 60), ("C#4", 61), ("C#", 61), ("Cb", 59)),
)
def test_pitch_parse(string: str, pitch_midi_interval: int):
    assert Pitch.parse(string) == Pitch(pitch_midi_interval)


@pytest.mark.parametrize(
    ("string", "index"), (("A", 0), ("A#", 1), ("Bb", 1), ("C", 3))
)
def test_note_name_index(string: str, index: int):
    assert NoteName.parse(string).index == index


def test_triad_pitched():
    assert [
        x.midi_interval for x in major.triad(1).pitched(Pitch.parse("C")).pitches
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
        NoteName.parse(string)


@pytest.mark.parametrize("string", ("aasdf", "X4", "XX", "CC4", "#", "aspdf9uh2"))
def test_parse_fail_pitch(string):
    with pytest.raises(ValueError):
        Pitch.parse(string)


def test_fast_api():
    with io.BytesIO() as file_object:
        make_midi(
            chord_progression=ChordProgression([1, 3], Pitch.parse("D#")),
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
    assert (
        TestClient(app)
        .get(
            "/",
            params=[
                ["chord_names", "A A C D"],
                ["tempo", 195],
            ],
        )
        .status_code
        == 200
    )
    assert (
        TestClient(app)
        .get(
            "/",
            params=[
                ["chord_names", "A A C D"],
                ["chord_numbers", 1],
                ["tempo", 195],
            ],
        )
        .status_code
        == 422
    )


def test_sequence():
    assert MidiChord(1, [Pitch.parse("C")]) * 2 == BuilderValueSequence(
        [
            BuilderValue.make(MidiChord(1, [Pitch.parse("C")])),
            BuilderValue.make(MidiChord(1, [Pitch.parse("C")])),
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


@pytest.mark.parametrize(
    ("chord_name_string", "value"),
    (("c", ChordName(NoteName(Letter.C, Accidental.natural), ChordType.major, [])),),
)
def test_parse_chord_name_value(chord_name_string: str, value: ChordName):
    assert ChordName.parse(chord_name_string) == value


@pytest.mark.parametrize(
    ("chord_name_string", "notes"),
    (
        (
            "c",
            {
                NoteName(Letter.C, Accidental.natural),
                NoteName(Letter.E, Accidental.natural),
                NoteName(Letter.G, Accidental.natural),
            },
        ),
    ),
)
def test_parse_chord_name_notes(chord_name_string: str, notes: Set[NoteName]):
    assert set(ChordName.parse(chord_name_string).notes) == notes


@pytest.mark.parametrize(
    ("chord_name_string", "note_strings"),
    (
        ("c", {"c", "e", "g"}),
        ("C", {"c", "e", "g"}),
        ("D#", {"d#", "g", "a#"}),
        ("Dmaj7", {"d", "f#", "a", "c#"}),
        ("Ddim", {"d", "f", "ab", "b"}),
        ("Ddim/G", {"d", "f", "ab", "b", "g"}),
        ("Ddimn7/G", {"d", "f", "ab", "b", "g", "c#"}),
    ),
)
def test_parse_chord_name_as_notes(chord_name_string: str, note_strings: Set[str]):
    assert set(ChordName.parse(chord_name_string).notes) == {
        NoteName.parse(note_string) for note_string in note_strings
    }


def test_next_note():
    assert NoteName.parse("e").next == NoteName.parse("f")


@pytest.mark.parametrize(
    ("root", "steps", "result"),
    (
        ("e", 1, "f"),
        ("c", 0, "c"),
        ("c", 1, "c#"),
        ("c", 2, "d"),
        ("c", 3, "d#"),
        ("c", 4, "e"),
        ("c", 5, "f"),
        ("c", 6, "f#"),
        ("c", 7, "g"),
        ("c", 8, "g#"),
        ("c", 9, "a"),
        ("c", 10, "a#"),
        ("c", 11, "b"),
        ("c", 12, "c"),
        ("c", -1, "b"),
        ("c", -2, "bb"),
    ),
)
def test_note_math(root: str, steps: int, result: str):
    assert NoteName.parse(root) + steps == NoteName.parse(result)


@pytest.mark.parametrize(
    ("letter", "result"),
    (
        ("a", True),
        ("b", False),
        ("c", True),
        ("d", True),
        ("e", False),
        ("f", True),
        ("g", True),
    ),
)
def test_has_sharp(letter: Letter, result: bool):
    assert Letter.parse(letter).has_sharp is result


@pytest.mark.parametrize(
    ("letter", "result"),
    (
        ("a", True),
        ("b", True),
        ("c", False),
        ("d", True),
        ("e", True),
        ("f", False),
        ("g", True),
    ),
)
def test_has_flat(letter: str, result: bool):
    assert Letter.parse(letter).has_flat is result


def test_note_eq():
    assert {NoteName.parse("g#")} == {NoteName.parse("ab")}


def test_accidental_int():
    assert int(Accidental.flat) == -1
    assert int(Accidental.natural) == 0
    assert int(Accidental.sharp) == 1


def test_chord_names_midi():
    with open(Path(__file__).parent / "chords.midi", "wb") as fn:
        make_midi(
            chord_progression=ChordNameProgression.parse(
                "fmaj7 fmaj7 g13 g13 gm7 f#7b5 fmaj7 f#7b5 " * 2
                + "F#maj7 " * 2
                + "B9 " * 2
                + "F#m7 " * 2
                + "D9 " * 2
                + "Gm7 " * 2
                + "Eb9 " * 2
                + "Am7 Abm7 Gm7 F#7b5 "
                + "fmaj7 fmaj7 g13 g13 gm7 f#7b5 fmaj7 f#7b5"
            ),
            file_object=fn,
        )


@pytest.mark.parametrize(
    ("class_type", "string"),
    (
        (Accidental, "asdf"),
        (NoteName, "asdf"),
        (BassNoteName, "G"),
        (NoteName, "#"),
    ),
)
def test_bad_parsing(class_type: type, string: str):
    with pytest.raises(ValueError):
        print(class_type.parse(string))


@pytest.mark.parametrize(
    ("class_type", "string", "value"),
    ((Octave, "", Octave()),),
)
def test_parse_default(class_type: type, string: str, value):
    assert class_type.parse(string) == value
