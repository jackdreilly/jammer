"""Jammer tool and server, with midi export and song-builder"""
from __future__ import annotations

import io
import itertools
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import _MISSING_TYPE, Field, asdict, dataclass, field, fields
from enum import Enum, auto, unique
from typing import (
    Any,
    BinaryIO,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Union,
)

import fastapi
from fastapi import HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.params import Depends, Query
from fastapi.responses import FileResponse
from midiutil.MidiFile import MIDIFile
from pydantic import BaseModel
from pydantic.types import ConstrainedInt, conint, constr

__version__ = "0.3.0"


@dataclass(frozen=True)
class DataclassFieldType:
    type: Any

    @classmethod
    def make(cls, field_type: str):
        if isinstance(field_type, str):
            field_type = eval(field_type)
        return cls(field_type)

    @property
    def is_list(self) -> bool:
        return getattr(self.type, "_name", None) == "List"

    @property
    def is_literal(self) -> bool:
        return getattr(self.type, "__origin__", None) is Literal

    @property
    def is_int(self) -> bool:
        return isinstance(self.type, type) and issubclass(
            self.type, (int, ConstrainedInt)
        )

    @property
    def list_type(self) -> DataclassFieldType:
        return self.make(self.type.__args__[0])

    @property
    def literal_value(self):
        return self.type.__args__[0]

    @property
    def is_dataclass(self) -> bool:
        return isinstance(self.type, type) and issubclass(
            self.type, DataclassParserMixin
        )


@dataclass(frozen=True)
class DataclassField:
    dataclass_field: Field

    @property
    def type(self) -> DataclassFieldType:
        return DataclassFieldType.make(self.dataclass_field.type)

    @property
    def name(self) -> str:
        return self.dataclass_field.name

    @property
    def default(self):
        # if self.type.is_literal:
        #     return self.type.literal_value
        return self.dataclass_field.default

    @property
    def has_default(self):
        return not isinstance(self.default, _MISSING_TYPE)


class DataclassParserMixin:
    @dataclass
    class ParserCounter:
        count: int = 0

        def pattern_name(self, name: str) -> int:
            self.count += 1
            return f"{name}{self.count}"

        @property
        def copied(self) -> DataclassParserMixin.ParserCounter:
            return DataclassParserMixin.ParserCounter(self.count)

    @classmethod
    def pattern(cls, counter: ParserCounter = None, *_, **__) -> re.Pattern:
        counter = counter or cls.ParserCounter()

        def field_pattern(field_type: DataclassFieldType) -> re.Pattern:
            if field_type.is_literal:
                return re.compile(re.escape(field_type.literal_value))
            if field_type.is_list:
                return re.compile(f"({field_pattern(field_type.list_type).pattern})*")
            if field_type.is_int:
                return re.compile(r"\d*")
            if field_type.is_dataclass:
                return field_type.type.pattern(counter)
            return field_type.type.pattern()

        return re.compile(
            "".join(
                f"(?P<{counter.pattern_name(field_.name)}>{field_pattern(field_.type).pattern}){'?' if field_.has_default else ''}"
                for field_ in map(DataclassField, fields(cls))
            )
        )

    @classmethod
    def parse(cls, string: str, counter: ParserCounter = None, *_, **__):
        counter = counter or cls.ParserCounter()
        outer_pattern = cls.pattern(counter.copied)
        outer_match = outer_pattern.fullmatch(string)

        def field_value(
            field_name: str,
            field_type: DataclassFieldType,
            match: re.Match,
            default_value=None,
        ) -> re.Pattern:
            if not match:
                raise ValueError("No match")
            pattern_name = counter.pattern_name(field_name)
            value = match.groupdict().get(pattern_name)
            if field_type.is_literal:
                return field_type.literal_value
            if field_type.is_list:
                return [
                    field_type.list_type.type.parse(list_match.group(0))
                    for list_match in list(
                        field_type.list_type.type.pattern(counter).finditer(value)
                    )
                    if list_match.group(0)
                ]
            if value is None:
                return default_value
            if field_type.is_int:
                if not value:
                    value = default_value
                return int(value)
            if field_type.is_dataclass:
                return field_type.type.parse(value, counter)
            return field_type.type.parse(value)

        return cls(
            **{
                field_.name: field_value(
                    field_.name,
                    field_.type,
                    outer_match,
                    default_value=field_.default,
                )
                for field_ in map(DataclassField, fields(cls))
            }
        )


@unique
class AutoEnum(str, Enum):
    """Generates enum value from enum name"""

    def _generate_next_value_(self: str, *_, **__):
        return self


class ParserMixin:
    @classmethod
    def string_mapping(cls) -> Dict:
        return {k: record for record in cls for k in (record.name, record.name.lower())}

    @classmethod
    def pattern(cls) -> re.Pattern:
        return re.compile("(" + "|".join(map(re.escape, cls.string_mapping())) + ")")

    @classmethod
    def parse(cls, string: str):
        mapping = cls.string_mapping()
        if value := mapping.get(string):
            return value
        raise ValueError(f"{string} not valid enum string for {cls} ({mapping})")


@dataclass(frozen=True)
class Octave(DataclassParserMixin):
    octave_value: int = 4

    def __mul__(self, i: int) -> Octave:
        return Octave(self.octave_value * i)

    def __add__(self, i: int) -> Octave:
        return Octave(self.octave_value + i)

    def __sub__(self, i: int) -> Octave:
        return self + (-i)

    def __rmul__(self, i: int) -> Octave:
        return self * i

    def __int__(self) -> int:
        return self.octave_value * 12


OCTAVE = Octave(1)


class Letter(ParserMixin, AutoEnum):
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


class AccidentalMixin(ParserMixin):
    @classmethod
    def string_mapping(cls) -> Dict:
        return {
            "b": cls.flat,
            "n": cls.natural,
            "#": cls.sharp,
            **{k.value: k for k in cls},
        }

    def __int__(self) -> int:
        return list(type(self)).index(self) - 1


class RequiredAccidental(AccidentalMixin, AutoEnum):
    """Musical accidental"""

    flat = "♭"
    natural = "♮"
    sharp = "♯"

    @property
    def normed(self) -> Accidental:
        return Accidental[self.name]


class Accidental(AccidentalMixin, AutoEnum):
    """Musical accidental w/ optional natural"""

    flat = "♭"
    natural = "♮"
    sharp = "♯"

    @classmethod
    def string_mapping(cls) -> Dict:
        return {
            **{k: v.normed for k, v in RequiredAccidental.string_mapping().items()},
            "": cls.natural,
        }


@dataclass(frozen=True)
class NoteName(DataclassParserMixin):
    letter: Letter
    accidental: Accidental = Accidental.natural

    def augmented(self, accidental: Accidental) -> NoteName:
        return self + int(accidental)

    @property
    def is_natural(self) -> bool:
        return self.accidental == Accidental.natural

    @property
    def is_flat(self) -> bool:
        return self.accidental == Accidental.flat

    @property
    def is_sharp(self) -> bool:
        return self.accidental == Accidental.sharp

    @property
    def next(self) -> NoteName:
        if self.is_flat:
            return self.naturalized
        if self.is_natural and self.letter.has_sharp:
            return self.sharped
        return NoteName(
            self.letter + 1,
            Accidental.natural
            if self.letter.has_sharp or self.is_natural
            else Accidental.sharp,
        )

    @property
    def naturalized(self) -> NoteName:
        return NoteName(self.letter, Accidental.natural)

    @property
    def flattened(self) -> NoteName:
        return NoteName(self.letter, Accidental.flat)

    @property
    def sharped(self) -> NoteName:
        return NoteName(self.letter, Accidental.sharp)

    @property
    def previous(self) -> NoteName:
        if self.is_sharp:
            return self.naturalized
        if self.is_natural and self.letter.has_flat:
            return self.flattened
        return NoteName(
            self.letter - 1,
            Accidental.natural
            if self.letter.has_flat or self.is_natural
            else Accidental.flat,
        )

    def __add__(self, steps: int) -> NoteName:
        if not steps:
            return self
        if steps > 0:
            return self.next + (steps - 1)
        return self.previous + (steps + 1)

    @property
    def normed(self) -> NoteName:
        return (
            self
            if self.is_natural
            else NoteName(
                self.letter - 1,
                Accidental.sharp if self.letter.has_flat else Accidental.natural,
            )
            if self.is_flat
            else self
            if self.letter.has_sharp
            else NoteName(self.letter + 1, Accidental.natural)
        )

    def __eq__(self, other: NoteName) -> bool:
        return asdict(self.normed) == asdict(other.normed)

    def __hash__(self) -> int:
        return hash(str(self.normed))

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
class NoteNameOctave(DataclassParserMixin):
    note_name: NoteName
    octave: Octave


@dataclass(frozen=True)
class Pitch:
    midi_interval: int

    _A4_MIDI_INTERVAL = 57
    _A4_OCTAVE = 4

    @classmethod
    def pattern(cls) -> re.Pattern:
        return NoteNameOctave.pattern()

    @classmethod
    def parse(cls, pitch_string: str, *args, **kwargs) -> Pitch:
        parsed: NoteNameOctave = NoteNameOctave.parse(pitch_string, *args, **kwargs)
        return cls.from_note(parsed.note_name, parsed.octave)

    def __add__(self, interval: int) -> Pitch:
        return Pitch(self.midi_interval + int(interval))

    def __sub__(self, interval: int) -> Pitch:
        return self + (-1 * interval)

    @classmethod
    def from_note(cls, note_name: NoteName, octave: Octave) -> Pitch:
        octave = octave if isinstance(octave, Octave) else Octave(octave)
        return cls(
            cls._A4_MIDI_INTERVAL + int(octave - cls._A4_OCTAVE) + note_name.index
        )

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

    def keyed(self, key: NoteName) -> KeyScale:
        return KeyScale(key, self)

    def __iter__(self) -> Iterator[int]:
        return iter(self.intervals)

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

    def __getitem__(self, interval: int) -> NoteName:
        return next(itertools.islice(self, interval - 1, None))

    def __iter__(self) -> Iterator[NoteName]:
        for interval in self.scale:
            yield self.key + interval


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
    key: Pitch = Pitch.parse("c")
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
class BuilderValue(BuildValueMixin):
    value: BuilderValueType
    advances_time: bool = True

    def using(self, advances_time: bool = None) -> BuilderValue:
        return (
            self
            if advances_time is None or self.advances_time is advances_time
            else BuilderValue(self.value, advances_time)
        )

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

    @classmethod
    def make(
        cls, value: Union[BuilderValue, BuilderValueType], advances_time: bool = None
    ) -> BuilderValue:
        return (
            value.using(advances_time=advances_time)
            if isinstance(value, BuilderValue)
            else BuilderValue(
                BuilderValueSequence(list(value)),
                advances_time if advances_time is not None else True,
            )
            if isinstance(value, Iterable)
            else BuilderValue(
                value,
                advances_time if advances_time is not None else True,
            )
        )


@dataclass(frozen=True)
class BuilderValueSequence:
    sequence: List[BuilderValue]

    def __mul__(self, repeat: int) -> BuilderValueSequence:
        return BuilderValueSequence([x for _ in range(repeat) for x in self.sequence])

    def __rshift__(self, value: BuilderValueType):
        return BuilderValueSequence([*self.sequence, BuilderValue.make(value)])

    def __lshift__(self, value: BuilderValueType):
        return BuilderValue(self, False) >> BuilderValue.make(value)

    def __iter__(self):
        return iter(self.sequence)

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

    @contextmanager
    def reset_time(self, builder_value: BuilderValue):
        original_time = self.time
        yield
        if not builder_value.advances_time:
            self.time = original_time

    def __lshift__(self, builder_value: BuilderValue) -> MidiTrackPlayBuilder:
        return self >> BuilderValue.make(builder_value).using(advances_time=False)

    def __rshift__(
        self, builder_value_input: Union[MidiChord, Measure, float]
    ) -> MidiTrackPlayBuilder:
        builder_value = BuilderValue.make(builder_value_input)
        with self.reset_time(builder_value):
            if isinstance(builder_value.value, BuilderValueSequence):
                for sequence_value in builder_value.value:
                    self >>= sequence_value
                return self
            for chord_duration in builder_value.chord_durations:
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
            << (
                (
                    drummer.bass(2)
                    << ((drummer.ride() >> drummer.ride(2 / 3) >> drummer.ride(1 / 3)))
                )
                * 2
            )
            >> Rest(2)
            >> (2 / 3)
            >> drummer.snare()
            >> drummer.snare(1 / 3)
        )
        (
            piano
            >> MidiChord(duration=1 + 2 / 3, pitches=list(chord.pitches))
            >> MidiChord(duration=1 / 6, pitches=list(chord.pitches))
            >> Measure()
        )
        (
            bass
            >> (
                MidiChord.pitch(pitch - 2 * OCTAVE)
                for pitch in itertools.islice(itertools.cycle(chord), 4)
            )
        )

    MidiSong(
        tracks=[track.compiled for track in [piano, bass, drums]], tempo=tempo
    ).write_to(file_object)


app = fastapi.FastAPI()


class ChordProgressionModel(BaseModel):
    chord_numbers: List[ChordNumber]
    key: constr(regex=Pitch.pattern().pattern)
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


class ChordType(ParserMixin, Enum):
    diminished = [0, 3, 6, 9]
    diminished_seventh = [*diminished, 10]
    augmented = [0, 4, 8]
    augmented_seventh = [*augmented, 10]
    suspended = [0, 5, 7]
    suspended_two = [0, 2, 7]
    major = [0, 4, 7]
    two = sorted({*major, 2})
    minor = [0, 3, 7]
    dominant = [*major, 10]
    minor_seventh = [*minor, 10]
    major_seventh = [*major, 11]
    sixth = sorted({*major, 9})
    minor_sixth = sorted({*minor, 9})
    thirteenth = [*sixth, 10]
    minor_thirteenth = [*minor_sixth, 10]
    minor_two = sorted({*minor, 2})
    major_two = sorted({*major, 2})
    ninth = sorted({*major_two, 10})
    minor_ninth = sorted({*minor_two, 10})

    @classmethod
    def string_mapping(cls) -> Dict:
        return {
            "°": cls.diminished,
            "ø": cls.diminished,
            "ø7": cls.diminished_seventh,
            "dim": cls.diminished,
            "dim7": cls.diminished_seventh,
            "+": cls.augmented,
            "aug": cls.augmented,
            "+7": cls.augmented_seventh,
            "aug7": cls.augmented_seventh,
            "7": cls.dominant,
            "m": cls.minor,
            "min": cls.minor,
            "m7": cls.minor_seventh,
            "min7": cls.minor_seventh,
            "maj": cls.major,
            "maj7": cls.major_seventh,
            "": cls.major,
            "sus": cls.suspended,
            "sus2": cls.suspended_two,
            "2": cls.two,
            "9": cls.ninth,
            "6": cls.sixth,
            "13": cls.thirteenth,
            "m6": cls.minor_sixth,
            "min6": cls.minor_sixth,
            "m2": cls.minor_two,
            "min2": cls.minor_two,
            "m9": cls.minor_ninth,
            "min9": cls.minor_ninth,
            "m13": cls.minor_thirteenth,
            "min13": cls.minor_thirteenth,
            "maj2": cls.major_two,
        }

    def __iter__(self) -> Iterator[int]:
        return iter(self.value)


@dataclass(frozen=True)
class ChordAccidental(DataclassParserMixin):
    accidental_: RequiredAccidental
    interval: ChordNumber

    @property
    def accidental(self) -> Accidental:
        return self.accidental_.normed


@dataclass(frozen=True)
class BassNoteName(DataclassParserMixin):
    slash: Literal["/"]
    note_name: NoteName


@dataclass(frozen=True)
class ChordName(DataclassParserMixin):
    root: NoteName
    chord_type: ChordType
    accidentals: List[ChordAccidental] = field(default_factory=list)
    bass: BassNoteName = None

    @property
    def scale(self) -> KeyScale:
        return major.keyed(self.root)

    @property
    def notes(self) -> Set[NoteName]:
        return sorted(
            {
                *([self.bass.note_name] if self.bass else []),
                *{
                    self.scale[accidental.interval].augmented(accidental.accidental)
                    for accidental in self.accidentals
                },
                *{self.root + step for step in self.chord_type},
            },
            key=lambda x: {self.bass and self.bass.note_name: 0, self.root: 1}.get(
                x, 2
            ),
        )

    @property
    def midi_friendly(self) -> ChordNameMidiFriendly:
        return ChordNameMidiFriendly(self)


@dataclass(frozen=True)
class ChordNameMidiFriendly:
    chord: ChordName

    @property
    def pitches(self) -> Iterable[Pitch]:
        for note in self.chord.notes:
            yield Pitch.from_note(note, Octave(4))

    def __iter__(self):
        yield from self.pitches


@dataclass(frozen=True)
class ChordNameProgression:
    chords_: List[ChordName]

    @classmethod
    def pattern(cls) -> re.Pattern:
        parser_counter = ChordName.ParserCounter()
        pattern_1 = ChordName.pattern(parser_counter)
        pattern_2 = ChordName.pattern(parser_counter)
        return re.compile(rf"{pattern_1.pattern}(\s+{pattern_2.pattern})*")

    @classmethod
    def parse(cls, chord_names_string: str) -> ChordNameProgression:
        return cls(
            list(
                ChordName.parse(chord_name_string.strip())
                for chord_name_string in chord_names_string.split(" ")
            )
        )

    @property
    def chords(self) -> Generator[ChordNameMidiFriendly, None, None]:
        for chord in self.chords_:
            yield chord.midi_friendly


@app.get("/", response_class=FileResponse)
def get_midi(
    chord_numbers: Optional[List[ChordNumber]] = Query(None),
    chord_names: Optional[
        constr(regex=f"^{ChordNameProgression.pattern().pattern}$")
    ] = Query(None),
    key: constr(regex=f"^{Pitch.pattern().pattern}$") = Query("C"),
    tempo: Tempo = 150,
    temp_file=Depends(create_temp_file),
):
    if chord_numbers and chord_names:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Cannot only supply one of chord_numbers or chord_names",
        )
    file_object, path = temp_file
    make_midi(
        chord_progression=ChordProgression(chord_numbers, Pitch.parse(key), major)
        if chord_numbers
        else ChordNameProgression.parse(chord_names),
        file_object=file_object,
        tempo=tempo,
    )
    file_object.flush()
    return FileResponse(path, filename="jammer.midi")
