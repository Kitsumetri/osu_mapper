"""Hermetic tests for the osu!.db parser (synthetic database, no real client)."""
import struct

from src.data.osu_db import (
    RANKED_STATUSES,
    STATUS_PENDING,
    STATUS_RANKED,
    parse_osu_db,
    ranked_osu_paths,
)

DB_VERSION = 20260612


def _str(s: str) -> bytes:
    if not s:
        return b"\x00"
    raw = s.encode("utf-8")
    # ULEB128 length (small values only in tests -> single byte is enough)
    assert len(raw) < 128
    return b"\x0b" + bytes([len(raw)]) + raw


def _star_list(pairs: list[tuple[int, float]]) -> bytes:
    out = struct.pack("<i", len(pairs))
    for mod, rating in pairs:
        out += b"\x08" + struct.pack("<i", mod)        # typed Int (mod)
        out += b"\x0c" + struct.pack("<f", rating)     # typed Single (rating)
    return out


def _beatmap(folder: str, osu_filename: str, status: int, mode: int,
             set_id: int = 1, beatmap_id: int = 1) -> bytes:
    b = b""
    for s in ("Artist", "ArtistU", "Title", "TitleU", "Creator", "Diff",
              "audio.mp3", "deadbeef", osu_filename):
        b += _str(s)
    b += bytes([status])
    b += struct.pack("<HHH", 10, 5, 1)                 # circles/sliders/spinners
    b += struct.pack("<q", 0)                          # last modification
    b += struct.pack("<ffff", 9.0, 4.0, 5.0, 7.0)      # AR/CS/HP/OD
    b += struct.pack("<d", 1.4)                        # slider velocity
    for _ in range(4):                                 # 4 star-rating lists
        b += _star_list([(0, 5.5)])
    b += struct.pack("<iii", 60, 120000, -1)           # drain/total/preview
    b += struct.pack("<i", 1)                          # one timing point
    b += struct.pack("<ddB", 300.0, 0.0, 1)            # bpm/offset/inherited
    b += struct.pack("<iii", beatmap_id, set_id, 0)    # beatmap/set/thread id
    b += bytes([0, 0, 0, 0])                           # grades
    b += struct.pack("<h", 0)                          # local offset
    b += struct.pack("<f", 0.7)                        # stack leniency
    b += bytes([mode])                                 # game mode
    b += _str("source") + _str("tags")
    b += struct.pack("<h", 0)                          # online offset
    b += _str("font")
    b += bytes([1])                                    # is unplayed
    b += struct.pack("<q", 0)                          # last played
    b += bytes([0])                                    # is osz2
    b += _str(folder)
    b += struct.pack("<q", 0)                          # last checked vs repo
    b += bytes([0, 0, 0, 0, 0])                        # ignore/disable flags
    b += struct.pack("<i", 0)                          # last modification (again)
    b += bytes([0])                                    # mania scroll speed
    return b


def _db(beatmaps: list[bytes]) -> bytes:
    head = struct.pack("<i", DB_VERSION)
    head += struct.pack("<i", 1)        # folder count
    head += bytes([1])                  # account unlocked
    head += struct.pack("<q", 0)        # unlock date
    head += _str("Tester")              # player name
    head += struct.pack("<i", len(beatmaps))
    body = b"".join(beatmaps)
    tail = struct.pack("<i", 0)         # user permissions
    return head + body + tail


def _write_db(tmp_path, beatmaps):
    p = tmp_path / "osu!.db"
    p.write_bytes(_db(beatmaps))
    return p


def test_parses_fields_and_consumes_fully(tmp_path):
    db = _write_db(tmp_path, [
        _beatmap("123 Song A", "a.osu", STATUS_RANKED, 0, set_id=123),
        _beatmap("456 Song B", "b.osu", STATUS_PENDING, 0, set_id=456),
    ])
    maps = parse_osu_db(db)
    assert len(maps) == 2
    assert maps[0].folder == "123 Song A"
    assert maps[0].osu_filename == "a.osu"
    assert maps[0].status == STATUS_RANKED
    assert maps[0].is_ranked
    assert maps[0].set_id == 123
    assert not maps[1].is_ranked


def test_ranked_statuses_membership():
    assert STATUS_RANKED in RANKED_STATUSES
    assert STATUS_PENDING not in RANKED_STATUSES


def test_ranked_osu_paths_joins_to_disk(tmp_path):
    songs = tmp_path / "Songs"
    (songs / "123 Song A").mkdir(parents=True)
    (songs / "123 Song A" / "a.osu").write_text("x", encoding="utf-8")
    # b.osu (pending) and c.osu (ranked but missing on disk) must be excluded
    db = _write_db(tmp_path, [
        _beatmap("123 Song A", "a.osu", STATUS_RANKED, 0),
        _beatmap("123 Song A", "b.osu", STATUS_PENDING, 0),
        _beatmap("999 Gone", "c.osu", STATUS_RANKED, 0),
        _beatmap("123 Song A", "mania.osu", STATUS_RANKED, 3),  # wrong mode
    ])
    paths = ranked_osu_paths(songs, db)
    assert paths == {(songs / "123 Song A" / "a.osu").resolve()}
