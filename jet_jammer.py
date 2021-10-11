"""Jammer tool and server, with midi export and song-builder"""
from __future__ import annotations

import io
import itertools
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum, auto, unique
from typing import (
    BinaryIO,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
)

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.params import Depends, Query
from fastapi.responses import FileResponse
from midiutil.MidiFile import MIDIFile
from pydantic import BaseModel
from pydantic.types import conint, constr

__version__ = "0.2.8"


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


class MidiInstrument(Enum):
    """Midi program number for instrument"""

    piano = 0
    bass = 32


class MidiDrum(Enum):
    """Midi pitch number for drum sounds"""

    bass = 35
    snare = 38
    ride = 51

    @property
    def chord(self) -> List[Pitch]:
        return [Pitch(self.value)]


class MidiChannel(Enum):
    """Midi channel number for special instruments"""

    drum = 9


class BuildValueMixin:
    def __mul__(self, repeat: int) -> BuilderValueSequence:
        return BuilderValueSequence.singleton(self) * repeat

    def __rshift__(self, value) -> BuilderValueSequence:
        return BuilderValueSequence.singleton(self) >> value

    def __lshift__(self, value) -> BuilderValueSequence:
        return BuilderValueSequence.singleton(self) << value


@dataclass(frozen=True)
class Measure(BuildValueMixin):
    count: int = 1

    def __mul__(self, i: int):
        return Measure(self.count * i)

    def __rmul__(self, i: int):
        return self * i

    def __int__(self) -> int:
        return self.count

    def duration(self, time: float, measure_duration: int) -> float:
        extra = -(time % measure_duration)
        offset = int(self) * measure_duration
        return extra + offset


@dataclass(frozen=True)
class Rest(BuildValueMixin):
    duration: float = 1


@dataclass(frozen=True)
class MidiChord(BuildValueMixin):
    """A single chord (or note) played on a MidiTrack"""

    duration: float
    pitches: List[Pitch]
    volume: int = None

    def at(self, time: float) -> MidiChordTimed:
        return MidiChordTimed(self, time)

    @classmethod
    def pitch(cls, note: Pitch, duration: float = 1) -> MidiChord:
        return cls(duration, [note])


BuilderValueType = Union[MidiChord, float, Measure, Rest, "BuilderValueSequence"]


class ChordDuration(Protocol):
    @property
    def midi_chord(self) -> Optional[MidiChord]:
        ...  # pragma: no cover

    def get_duration(self, time: float, measure_duration: int) -> float:
        ...  # pragma: no cover

    @classmethod
    def chord(cls, chord: MidiChord) -> ChordDuration:
        return ChordDurationChord(chord)

    @classmethod
    def rest(cls, duration: float) -> ChordDuration:
        return ChordDurationRest(duration)

    @classmethod
    def measure(cls, measure: Measure) -> ChordDuration:
        return ChordDurationMeasure(measure)


@dataclass(frozen=True)
class ChordDurationRest(ChordDuration):
    duration: float

    def get_duration(self, time: float, measure_duration: int) -> float:
        return self.duration


@dataclass(frozen=True)
class ChordDurationMeasure(ChordDuration):
    measure: Measure

    def get_duration(self, time: float, measure_duration: int) -> float:
        return self.measure.duration(time, measure_duration)


@dataclass(frozen=True)
class ChordDurationChord(ChordDuration):
    midi_chord_: MidiChord

    @property
    def midi_chord(self) -> Optional[MidiChord]:
        return self.midi_chord_

    def get_duration(self, time: float, measure_duration: int) -> float:
        return self.midi_chord.duration


@dataclass(frozen=True)
class BuilderValue:
    value: BuilderValueType

    @property
    def chord_durations(self) -> Generator[ChordDuration, None, None]:
        if isinstance(self.value, MidiChord):
            yield ChordDuration.chord(self.value)
        if isinstance(self.value, float):
            yield ChordDuration.rest(self.value)
        if isinstance(self.value, Rest):
            yield ChordDuration.rest(self.value.duration)
        if isinstance(self.value, Measure):
            yield ChordDuration.measure(self.value)
        if isinstance(self.value, BuilderValueSequence):
            for value in self.value.sequence:
                yield from BuilderValue.make(value).chord_durations

    @classmethod
    def make(cls, value: Union[BuilderValue, BuilderValueType]) -> BuilderValue:
        return (
            value
            if isinstance(value, BuilderValue)
            else BuilderValue(BuilderValueSequence(list(value)))
            if isinstance(value, Iterable)
            else BuilderValue(value)
        )


@dataclass(frozen=True)
class BuilderValueSequence:
    sequence: List[BuilderValue]

    def __mul__(self, repeat: int) -> BuilderValueSequence:
        return BuilderValueSequence([x for _ in range(repeat) for x in self.sequence])

    def __rshift__(self, value: BuilderValueType):
        return BuilderValueSequence([*self.sequence, BuilderValue.make(value)])

    def __lshift__(self, value: BuilderValueType):
        value: BuilderValue = BuilderValue.make(value)
        sequence = self >> value
        if isinstance(value.value, MidiChord):
            if duration := value.value.duration:
                return sequence >> -1 * duration
        return sequence

    @classmethod
    def singleton(cls, builder_value: BuilderValue):
        return cls(
            [
                builder_value
                if isinstance(builder_value, BuilderValue)
                else BuilderValue(builder_value)
            ]
        )


@dataclass(frozen=True)
class MidiTrack:
    """Baseline information for Midi Track"""

    name: str
    volume: int = 100
    channel: int = None
    instrument: MidiInstrument = None


@dataclass(frozen=True)
class MidiTrackPlay:
    midi_track: MidiTrack
    midi_chords: List[MidiChordTimed]


@dataclass(frozen=True)
class MidiChordTimed:
    midi_chord: MidiChord
    time: float


class Drummer:
    def bass(self, duration: float = 1) -> MidiChord:
        return MidiChord(duration, MidiDrum.bass.chord)

    def ride(self, duration: float = 1) -> MidiChord:
        return MidiChord(duration, MidiDrum.ride.chord)

    def snare(self, duration: float = 1) -> MidiChord:
        return MidiChord(duration, MidiDrum.snare.chord)


drummer = Drummer()


@dataclass
class MidiTrackPlayBuilder:
    midi_track: MidiTrack
    midi_chords: List[MidiChordTimed] = field(default_factory=list)
    time: float = 0
    measure_duration: int = 4

    @property
    def measure(self) -> int:
        return self.time // self.measure_duration

    @measure.setter
    def measure(self, measure: int) -> int:
        self.time = measure * self.measure_duration

    @property
    @contextmanager
    def reset_time(self):
        original_time = self.time
        yield
        self.time = original_time

    def __lshift__(self, builder_value: BuilderValue) -> MidiTrackPlayBuilder:
        builder_value = BuilderValue.make(builder_value)
        with self.reset_time:
            return self >> builder_value

    def __rshift__(
        self, builder_value: Union[MidiChord, Measure, float]
    ) -> MidiTrackPlayBuilder:
        for chord_duration in BuilderValue.make(builder_value).chord_durations:
            if chord := chord_duration.midi_chord:
                self.midi_chords.append(chord.at(self.time))
            self.time += (
                chord_duration.get_duration(self.time, self.measure_duration) or 0
            )
        return self

    @property
    def compiled(self) -> MidiTrackPlay:
        return MidiTrackPlay(self.midi_track, list(self.midi_chords))


@dataclass(frozen=True)
class MidiSong:
    tracks: List[MidiTrackPlay]
    tempo: Tempo

    def write_to(self, file_object: io.BytesIO):
        midi_file = MIDIFile(numTracks=len(self.tracks))
        for track_number, track in enumerate(self.tracks):
            channel = track.midi_track.channel
            if channel is None:
                channel = track_number
            midi_file.addTrackName(
                track=track_number,
                time=0,
                trackName=track.midi_track.name,
            )
            midi_file.addTempo(
                track=track_number,
                time=0,
                tempo=self.tempo,
            )
            if program := track.midi_track.instrument:
                midi_file.addProgramChange(
                    tracknum=track_number,
                    channel=channel,
                    time=0,
                    program=program.value,
                )
            for chord in track.midi_chords:
                for pitch in chord.midi_chord.pitches:
                    midi_file.addNote(
                        track=track_number,
                        channel=channel,
                        pitch=pitch.midi_interval,
                        time=chord.time,
                        duration=chord.midi_chord.duration,
                        volume=chord.midi_chord.volume or track.midi_track.volume,
                    )
        midi_file.writeFile(fileHandle=file_object)


Tempo = conint(ge=60, le=300)


def make_midi(
    *, chord_progression: ChordProgression, tempo: int = 120, file_object: BinaryIO
):
    piano = MidiTrackPlayBuilder(
        midi_track=MidiTrack(name="Piano", instrument=MidiInstrument.piano)
    )
    bass = MidiTrackPlayBuilder(
        midi_track=MidiTrack(name="Bass", instrument=MidiInstrument.bass)
    )
    drums = MidiTrackPlayBuilder(
        midi_track=MidiTrack(name="Drums", channel=MidiChannel.drum.value)
    )

    for chord in chord_progression.chords:
        (
            drums
            << (drummer.bass(2) << drummer.bass(2) << Rest(0))
            << ((drummer.ride() >> drummer.ride(2 / 3) >> drummer.ride(1 / 3)) * 2)
            >> Rest(2)
            >> (2 / 3)
            >> drummer.snare()
            >> drummer.snare(1 / 3)
        )
        (
            piano
            >> MidiChord(duration=1 + 2 / 3, pitches=list(chord.pitches))
            >> MidiChord(duration=1 / 6, pitches=list(chord.pitches))
            >> 1 / 6
            >> (1 * Measure() * 1)
        )
        (bass >> (MidiChord.pitch(pitch - 2 * OCTAVE) for pitch in chord))

    MidiSong(
        tracks=[track.compiled for track in [piano, bass, drums]], tempo=tempo
    ).write_to(file_object)


app = fastapi.FastAPI()


class ChordProgressionModel(BaseModel):
    chord_numbers: List[ChordNumber]
    key: constr(regex=Pitch.regex.pattern)
    scale: Literal["major"] = "major"

    class Config(BaseModel.Config):
        extra = "forbid"


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


app.add_middleware(GZipMiddleware)


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
        file_object=file_object,
        tempo=tempo,
    )
    file_object.flush()
    return FileResponse(path, filename="jammer.midi")
