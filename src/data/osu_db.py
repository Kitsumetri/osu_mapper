"""Parse the osu! client database (``osu!.db``) to recover *ranked status*.

The ``.osu`` files in the Songs library don't record whether a beatmap is ranked
-- that's online metadata the client caches in ``osu!.db``. Ranked/approved/loved
maps are community-vetted (complete hitsounds, kiai, timing, sane patterns), so
filtering the training set to them is the single biggest data-quality lever.

This reads the binary ``osu!.db`` format and returns one record per beatmap with
the fields we need to join back to the library: the containing folder name, the
``.osu`` file name, the ranked status, the game mode, and the set id.

Format reference: https://osu.ppy.sh/wiki/en/Client/File_formats/Db_(file_format)
Tested against client version 20260612 (modern: float difficulty values, no
per-entry size prefix, star-rating pair lists present).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

# Ranked-status byte values (osu!.db).
STATUS_UNKNOWN = 0
STATUS_UNSUBMITTED = 1
STATUS_PENDING = 2  # pending / wip / graveyard
STATUS_UNUSED = 3
STATUS_RANKED = 4
STATUS_APPROVED = 5
STATUS_QUALIFIED = 6
STATUS_LOVED = 7

# Community-vetted statuses we treat as high quality for training.
RANKED_STATUSES = frozenset({STATUS_RANKED, STATUS_APPROVED, STATUS_LOVED})


@dataclass
class DbBeatmap:
    folder: str
    osu_filename: str
    status: int
    mode: int
    set_id: int
    beatmap_id: int

    @property
    def is_ranked(self) -> bool:
        return self.status in RANKED_STATUSES


class _Reader:
    """Little-endian cursor over the osu!.db byte buffer."""

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def _take(self, n: int) -> bytes:
        b = self.buf[self.pos:self.pos + n]
        if len(b) != n:
            raise EOFError(f"unexpected EOF at {self.pos} (+{n})")
        self.pos += n
        return b

    def byte(self) -> int:
        return self._take(1)[0]

    def short(self) -> int:
        return struct.unpack("<H", self._take(2))[0]

    def int(self) -> int:
        return struct.unpack("<i", self._take(4))[0]

    def long(self) -> int:
        return struct.unpack("<q", self._take(8))[0]

    def single(self) -> float:
        return struct.unpack("<f", self._take(4))[0]

    def double(self) -> float:
        return struct.unpack("<d", self._take(8))[0]

    def bool(self) -> bool:
        return self.byte() != 0

    def uleb128(self) -> int:
        result = shift = 0
        while True:
            b = self.byte()
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result
            shift += 7

    def string(self) -> str:
        marker = self.byte()
        if marker == 0x00:
            return ""
        if marker != 0x0B:
            raise ValueError(f"bad string marker {marker:#x} at {self.pos - 1}")
        length = self.uleb128()
        return self._take(length).decode("utf-8", errors="replace")

    def _typed_value(self) -> None:
        """Consume one type-tagged value (osu! serialises these in star lists).

        Type bytes seen: 0x08 Int, 0x0c Single, 0x0d Double. Modern clients
        (>= ~2025) write the star rating as a Single (0x0c); older ones use a
        Double (0x0d)."""
        t = self.byte()
        if t == 0x08:
            self.int()
        elif t == 0x0C:
            self.single()
        elif t == 0x0D:
            self.double()
        else:
            raise ValueError(f"unknown typed-value tag {t:#x} at {self.pos - 1}")

    def int_double_pairs(self) -> None:
        """Skip a star-rating list: Int count, then count * (mod, rating) pairs,
        each a type-tagged Int followed by a type-tagged Single/Double."""
        count = self.int()
        for _ in range(count):
            self._typed_value()  # mod combination (Int)
            self._typed_value()  # star rating (Single on modern clients)


def _read_beatmap(r: _Reader, version: int) -> DbBeatmap:
    r.string()   # artist
    r.string()   # artist unicode
    r.string()   # title
    r.string()   # title unicode
    r.string()   # creator
    r.string()   # difficulty (version name)
    r.string()   # audio filename
    r.string()   # md5
    osu_filename = r.string()
    status = r.byte()
    r.short()    # hitcircles
    r.short()    # sliders
    r.short()    # spinners
    r.long()     # last modification time
    # difficulty values (AR, CS, HP, OD): Single in >= 20140609, Byte before that
    read_diff = r.single if version >= 20140609 else r.byte
    for _ in range(4):
        read_diff()
    r.double()   # slider velocity
    if version >= 20140609:
        for _ in range(4):       # star ratings: std, taiko, ctb, mania
            r.int_double_pairs()
    r.int()      # drain time (s)
    r.int()      # total time (ms)
    r.int()      # preview time
    n_timing = r.int()
    for _ in range(n_timing):    # (double bpm, double offset, bool inherited)
        r.double()
        r.double()
        r.bool()
    beatmap_id = r.int()
    set_id = r.int()
    r.int()      # thread id
    for _ in range(4):           # grades std/taiko/ctb/mania
        r.byte()
    r.short()    # local offset
    r.single()   # stack leniency
    mode = r.byte()
    r.string()   # source
    r.string()   # tags
    r.short()    # online offset
    r.string()   # title font
    r.bool()     # is unplayed
    r.long()     # last played
    r.bool()     # is osz2
    folder = r.string()
    r.long()     # last checked against repo
    for _ in range(5):           # ignore sound/skin, disable sb/video, visual override
        r.bool()
    if version < 20140609:
        r.short()                # unknown (old versions only)
    r.int()      # last modification time (again)
    r.byte()     # mania scroll speed
    return DbBeatmap(folder=folder, osu_filename=osu_filename, status=status,
                     mode=mode, set_id=set_id, beatmap_id=beatmap_id)


def parse_osu_db(path: str | Path) -> list[DbBeatmap]:
    """Parse ``osu!.db`` into a list of :class:`DbBeatmap` records."""
    buf = Path(path).read_bytes()
    r = _Reader(buf)
    version = r.int()
    r.int()          # folder count
    r.bool()         # account unlocked
    r.long()         # unlock date
    r.string()       # player name
    n_beatmaps = r.int()
    maps = [_read_beatmap(r, version) for _ in range(n_beatmaps)]
    r.int()          # trailing user-permissions int (end of file)
    return maps


def ranked_osu_paths(songs_dir: str | Path, db_path: str | Path) -> set[Path]:
    """Return resolved paths of every ranked/approved/loved std ``.osu`` file.

    Joins osu!.db records (folder + filename + status) onto the on-disk library.
    Only ``mode == 0`` (osu!standard) ranked maps that actually exist are kept.
    """
    songs_dir = Path(songs_dir)
    paths: set[Path] = set()
    for bm in parse_osu_db(db_path):
        if bm.mode != 0 or not bm.is_ranked or not bm.osu_filename:
            continue
        p = songs_dir / bm.folder / bm.osu_filename
        if p.exists():
            paths.add(p.resolve())
    return paths
