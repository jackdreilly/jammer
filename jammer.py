from __future__ import annotations

import itertools
import re
from dataclasses import asdict, dataclass
from enum import Enum, auto, unique
from pathlib import Path
from typing import Generator, Iterable, List

from midiutil.MidiFile import MIDIFile


@unique
class AutoEnum(str, Enum):
    def _generate_next_value_(name: str, *args, **kwargs):
        return name


class Letter(AutoEnum):
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    G = auto()

    @property
    def has_sharp(self) -> bool:
        return self not in {self.B, self.E}

    @property
    def has_flat(self) -> bool:
        return (self - 1).has_sharp

    def __add__(self, i: int) -> Letter:
        letter_list = list(Letter)
        return letter_list[(letter_list.index(self) + i) % len(letter_list)]

    def __sub__(self, i: int) -> Letter:
        return self + (-i)

    @classmethod
    def from_string(cls, string: str) -> Letter:
        return cls(string.upper())


class Accidental(AutoEnum):
    flat = "♭"
    natural = "♮"
    sharp = "♯"

    @classmethod
    def from_string(cls, string: str) -> Accidental:
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

    @classmethod
    def from_string(cls, pitch_string: str) -> Pitch:
        match = re.match(NoteName.regex[:-1] + r"(?P<octave>\d?)$", pitch_string)
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
        return Pitch(self.midi_interval + interval)

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
        n = len(self.intervals)
        start_index = (number % n) - 1
        root = self.intervals[start_index]
        return Chord(
            root, [self.intervals[(i * 2 + start_index) % n] for i in range(1, 3)]
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


@dataclass(frozen=True)
class ChordProgression:
    chord_numbers: List[int]
    key: Pitch = Pitch.from_string("c")
    scale: Scale = major

    @property
    def chords(self) -> Iterable[KeyChord]:
        for chord_number in self.chord_numbers:
            yield self.scale.triad(chord_number).pitched(self.key)


def make_midi(chord_progression: ChordProgression, file: Path):
    midi_file = MIDIFile(1)
    track = 0  # the only track
    time = 0  # start at the beginning
    midi_file.addTrackName(track, time, "Sample Track")
    midi_file.addTempo(track, time, 120)
    channel = 0
    volume = 100
    pitch = 60  # C4 (middle C)
    time = 0  # start on beat 0
    duration = 4  # 1 beat long
    for measure, chord in enumerate(chord_progression.chords):
        for pitch in chord:
            midi_file.addNote(
                track, channel, pitch.midi_interval, measure * 4, duration, volume
            )

    # write it to disk
    with open(file, "wb") as outf:
        midi_file.writeFile(outf)
