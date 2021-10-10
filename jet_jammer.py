"""Jammer tool"""
from __future__ import annotations

import itertools
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum, auto, unique
from typing import BinaryIO, Generator, Iterable, List, Literal

import fastapi
import pydantic
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends, Query
from fastapi.responses import FileResponse
from midiutil.MidiFile import MIDIFile
from pydantic.types import conint, constr


@unique
class AutoEnum(str, Enum):
    """Generates enum value from enum name"""

    def _generate_next_value_(self: str, *_, **__):
        return self


@dataclass(frozen=True)
class Octave:
    count: int = 1

    def __mul__(self, i: int):
        return Octave(self.count * i)

    def __rmul__(self, i: int):
        return self * i

    def __int__(self) -> int:
        return self.count * 12


OCTAVE = Octave()


class Letter(AutoEnum):
    """Musical letter"""

    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    G = auto()

    @property
    def has_sharp(self) -> bool:
        """Has sharp black key"""
        return self not in {self.B, self.E}

    @property
    def has_flat(self) -> bool:
        """Has flat black key"""
        return (self - 1).has_sharp

    def __add__(self, i: int) -> Letter:
        """Increments letter"""
        letter_list = list(Letter)
        return letter_list[(letter_list.index(self) + i) % len(letter_list)]

    def __sub__(self, i: int) -> Letter:
        """Decrements letter"""
        return self + (-i)

    @classmethod
    def from_string(cls, string: str) -> Letter:
        """Parses letter from string"""
        return cls(string.upper())


class Accidental(AutoEnum):
    """Musical accidental"""

    flat = "♭"
    natural = "♮"
    sharp = "♯"

    @classmethod
    def from_string(cls, string: str) -> Accidental:
        """Parses accidental from string, supporting easy aliases"""
        return cls(
            {"b": cls.flat, "n": cls.natural, "#": cls.sharp, "": cls.natural}.get(
                string, string
            )
        )


@dataclass(frozen=True)
class NoteName:
    letter: Letter
    accidental: Accidental

    regex = r"^(?P<letter>[a-zA-Z])(?P<accidental>[♮♭#bn♯]?)$"

    @property
    def is_natural(self) -> bool:
        return self.accidental == Accidental.natural

    @property
    def is_flat(self) -> bool:
        return self.accidental == Accidental.flat

    def __eq__(self, other: NoteName) -> bool:
        def normalize(note_name: NoteName):
            return (
                note_name
                if note_name.is_natural
                else NoteName(
                    note_name.letter - 1,
                    Accidental.sharp
                    if note_name.letter.has_flat
                    else Accidental.natural,
                )
                if note_name.is_flat
                else note_name
                if note_name.letter.has_sharp
                else NoteName(note_name.letter + 1, Accidental.natural)
            )

        return asdict(normalize(self)) == asdict(normalize(other))

    @classmethod
    def natural(cls, letter: Letter) -> NoteName:
        return cls(letter, Accidental.natural)

    @classmethod
    def sharp(cls, letter: Letter) -> NoteName:
        return cls(letter, Accidental.sharp)

    @classmethod
    def iterator(cls):
        for letter in Letter:
            yield cls.natural(letter)
            if letter.has_sharp:
                yield cls.sharp(letter)

    @classmethod
    def from_a_offset(cls, offset: int) -> NoteName:
        return next(itertools.islice(cls.iterator(), offset, None))

    @property
    def index(self) -> int:
        return list(self.iterator()).index(self)

    @classmethod
    def from_string(cls, note_name_string: str) -> NoteName:
        match = re.match(cls.regex, note_name_string)
        if not match:
            raise ValueError(f'"{note_name_string}" not a NoteName')
        return NoteName(
            Letter.from_string(match.group("letter")),
            Accidental.from_string(
                match.group("accidental") or Accidental.natural.value
            ),
        )


@dataclass(frozen=True)
class Chord:
    root: int
    harmonics: List[int]

    @property
    def intervals(self) -> List[int]:
        return [self.root, *self.harmonics]

    def pitched(self, pitch: Pitch) -> KeyChord:
        return KeyChord(pitch.note_name, self, pitch.octave)


@dataclass(frozen=True)
class Pitch:
    midi_interval: int

    _A4_MIDI_INTERVAL = 57
    _A4_OCTAVE = 4

    regex = re.compile(NoteName.regex[:-1] + r"(?P<octave>\d?)$")

    @classmethod
    def from_string(cls, pitch_string: str) -> Pitch:
        match = cls.regex.match(pitch_string)
        if not match:
            raise ValueError(f'"{pitch_string}" not a Pitch')
        return cls.from_note(
            NoteName(
                Letter.from_string(match.group("letter")),
                Accidental.from_string(
                    match.group("accidental") or Accidental.natural.value
                ),
            ),
            int(match.group("octave") or 4),
        )

    def __add__(self, interval: int) -> Pitch:
        return Pitch(self.midi_interval + int(interval))

    def __sub__(self, interval: int) -> Pitch:
        return self + (-1 * interval)

    @classmethod
    def from_note(cls, note_name: NoteName, octave: int) -> Pitch:
        return cls(cls._A4_MIDI_INTERVAL + (octave - 4) * 12 + note_name.index)

    @property
    def a_offset(self) -> int:
        return (self.midi_interval - self._A4_MIDI_INTERVAL) % 12

    @property
    def note_name(self) -> NoteName:
        return NoteName.from_a_offset(self.a_offset)

    @property
    def octave(self) -> int:
        return (self.midi_interval - self._A4_MIDI_INTERVAL) // 12 + self._A4_OCTAVE


@dataclass(frozen=True)
class Scale:
    intervals: List[int]

    @classmethod
    def make(cls, *intervals: List[int]):
        return cls(sorted({0, *intervals}))

    def triad(self, number: int) -> Chord:
        interval_length = len(self.intervals)
        start_index = (number % interval_length) - 1
        root = self.intervals[start_index]
        return Chord(
            root,
            [
                self.intervals[(i * 2 + start_index) % interval_length]
                for i in range(1, 3)
            ],
        )

    def seventh(self, number: int) -> Chord:
        interval_length = len(self.intervals)
        start_index = (number % interval_length) - 1
        root = self.intervals[start_index]
        return Chord(
            root,
            [
                self.intervals[(i * 2 + start_index) % interval_length]
                for i in range(1, 4)
            ],
        )


@dataclass(frozen=True)
class KeyScale:
    key: NoteName
    scale: Scale
    octave: int = 4


@dataclass(frozen=True)
class KeyChord:
    key: NoteName
    chord: Chord
    octave: int = 4

    @property
    def pitches(self) -> Generator[Pitch, None, None]:
        for interval in self.chord.intervals:
            yield Pitch.from_note(self.key, self.octave) + interval

    def __iter__(self) -> Generator[Pitch, None, None]:
        return self.pitches


major = Scale.make(2, 4, 5, 7, 9, 11)

ChordNumber = conint(ge=1, le=7)


@dataclass(frozen=True)
class ChordProgression:
    chord_numbers: List[ChordNumber]
    key: Pitch = Pitch.from_string("c")
    scale: Scale = major

    @property
    def chords(self) -> Iterable[KeyChord]:
        for chord_number in self.chord_numbers:
            yield self.scale.seventh(chord_number).pitched(self.key)


def make_midi(
    *, chord_progression: ChordProgression, tempo: int = 120, file_object: BinaryIO
):
    midi_file = MIDIFile(3)
    midi_file.addTrackName(0, 0, "Piano")
    midi_file.addTempo(0, 0, tempo)
    midi_file.addTrackName(1, 0, "Bass")
    midi_file.addTempo(1, 0, tempo)
    midi_file.addTrackName(2, 0, "Drums")
    midi_file.addTempo(2, 0, tempo)
    midi_file.addProgramChange(0, 0, 0, 0)
    midi_file.addProgramChange(1, 1, 0, 32)
    time = 0
    for repeat in range(16):
        time = repeat * 4 * len(list(chord_progression.chords))
        for measure, chord in enumerate(chord_progression.chords):
            midi_file.addNote(2, 9, 35, time + measure * 4, 1, 100)
            midi_file.addNote(2, 9, 35, time + measure * 4 + 2, 1, 100)
            midi_file.addNote(2, 9, 51, time + measure * 4, 1, 100)
            midi_file.addNote(2, 9, 51, time + measure * 4 + 1, 1, 100)
            midi_file.addNote(2, 9, 51, time + measure * 4 + 1 + 2 / 3, 1, 100)
            midi_file.addNote(2, 9, 51, 2 + time + measure * 4, 1, 100)
            midi_file.addNote(2, 9, 51, 2 + time + measure * 4 + 1, 1, 100)
            midi_file.addNote(2, 9, 51, 2 + time + measure * 4 + 1 + 2 / 3, 1, 100)
            for pitch in chord:
                midi_file.addNote(
                    0,
                    0,
                    pitch.midi_interval,
                    time + measure * 4,
                    4 * 10 / 24,
                    100,
                )
                midi_file.addNote(
                    0,
                    0,
                    pitch.midi_interval,
                    time + measure * 4 + 4 * 10 / 24,
                    4 * 1 / 8,
                    100,
                )
            for i, pitch in enumerate(chord):
                pitch = pitch - 2 * OCTAVE
                midi_file.addNote(
                    1,
                    1,
                    pitch.midi_interval,
                    time + measure * 4 + i,
                    1,
                    100,
                )

    midi_file.writeFile(file_object)


app = fastapi.FastAPI()


class ChordProgressionModel(pydantic.BaseModel):
    chord_numbers: List[ChordNumber]
    key: constr(regex=Pitch.regex.pattern)
    scale: Literal["major"] = "major"


def create_temp_file():
    file_descriptor, path = tempfile.mkstemp(suffix=".midi")
    try:
        with os.fdopen(file_descriptor, "wb") as file_object:
            yield file_object, path
    finally:
        os.unlink(path)


app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Tempo = conint(ge=60, le=300)


@app.get("/", response_class=FileResponse)
def get_midi(
    chord_numbers: List[ChordNumber] = Query([1, 6, 2, 5]),
    key: constr(regex=Pitch.regex.pattern) = Query("C"),
    tempo: Tempo = 150,
    temp_file=Depends(create_temp_file),
):
    file_object, path = temp_file
    make_midi(
        chord_progression=ChordProgression(
            chord_numbers, Pitch.from_string(key), major
        ),
        tempo=tempo,
        file_object=file_object,
    )
    return FileResponse(path, filename="jammer.midi")
