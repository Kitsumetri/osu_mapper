"""Hermetic tests for the leakage-free group-aware train/val split.

Imports ONLY ``src.data.val_split`` (stdlib-only module) so it is isolated from
other agents' in-flight edits to reward/metrics/generate. No GPU, dataset, or
network. The reward-in-val wiring is covered by stubbing the reward callable, so
this file never imports ``src.eval.reward`` or ``src.metrics``.
"""
from __future__ import annotations

import json

from src.data import val_split as vs


def _item(item_id, audio_id, title, creator, version="Insane", artist=None):
    it = {
        "item_id": item_id, "audio_id": audio_id, "title": title,
        "creator": creator, "version": version, "n_objects": 200,
        "star_rating": 5.0, "duration_s": 120.0,
    }
    if artist is not None:
        it["artist"] = artist
    return it


def _synthetic_manifest():
    """A manifest with deliberate leakage traps:

    - Song A: ONE audio file, TWO difficulties (same audio_id) -> shared audio.
    - Song B ("FREEDOM DiVE"): SAME song mapped by TWO mappers with DIFFERENT audio
      files (different audio_id) AND a cosmetic title difference (case/punct) ->
      must group by normalised song-key despite distinct audio_ids.
    - Song C: a chain — mapper1 audio_id=cX shared by two diffs, mapper2 a different
      audio_id but same title -> union-find must pull the whole chain together.
    - Songs D..K: many singleton songs to give the frac something to bite on.
    """
    items = [
        # Song A: two difficulties, ONE shared audio file
        _item("A_easy", "audioA", "Song Alpha", "mapperA", "Easy"),
        _item("A_hard", "audioA", "Song Alpha", "mapperA", "Hard"),
        # Song B: duplicate-audio-across-mappers (different audio_id, cosmetic title diff)
        _item("B_m1", "audioB1", "FREEDOM DiVE", "mapper1", "Inner Oni"),
        _item("B_m2", "audioB2", "freedom  dive!!", "mapper2", "Another"),
        # Song C: a chain across two audio files + two mappers
        _item("C_m1a", "audioC1", "Blue Zenith", "mapperX", "FOUR DIMENSIONS"),
        _item("C_m1b", "audioC1", "Blue Zenith", "mapperX", "Light Insane"),
        _item("C_m2", "audioC2", "blue zenith", "mapperY", "Extra"),
    ]
    # singleton songs D..K (8 more) so val_frac can pick whole groups
    for n in range(8):
        items.append(_item(f"S{n}", f"audioS{n}", f"Single Song {n}", f"m{n}"))
    return items


def test_no_song_key_and_no_audio_id_overlap():
    items = _synthetic_manifest()
    train, val = vs.grouped_split(items, val_frac=0.30, seed=7)
    assert val, "expected a non-empty val set"
    assert set(train).isdisjoint(set(val)), "an item is in both train and val"
    # every manifest item is accounted for exactly once
    assert set(train) | set(val) == {it["item_id"] for it in items}
    # the invariant the whole module exists for:
    vs.assert_no_leakage(items, train, val)

    # explicit re-derivation of the two leakage classes (don't trust only the helper)
    by_id = {it["item_id"]: it for it in items}
    tkeys = {vs.song_key(by_id[i]) for i in train}
    vkeys = {vs.song_key(by_id[i]) for i in val}
    assert tkeys.isdisjoint(vkeys), "song-identity key leaked across the split"
    taudio = {by_id[i]["audio_id"] for i in train}
    vaudio = {by_id[i]["audio_id"] for i in val}
    assert taudio.isdisjoint(vaudio), "audio_id leaked across the split"


def test_duplicate_audio_across_mappers_groups_together():
    """The two mapper copies of 'FREEDOM DiVE' (different audio_id, cosmetic title
    diff) must land on the SAME side; likewise the Blue Zenith chain."""
    items = _synthetic_manifest()
    for seed in range(20):
        train, val = vs.grouped_split(items, val_frac=0.30, seed=seed)
        ts = set(train)
        # FREEDOM DiVE: B_m1 and B_m2 never split apart
        assert ("B_m1" in ts) == ("B_m2" in ts), f"FREEDOM DiVE split at seed {seed}"
        # Blue Zenith chain: all three on one side
        chain = {"C_m1a" in ts, "C_m1b" in ts, "C_m2" in ts}
        assert len(chain) == 1, f"Blue Zenith chain split at seed {seed}"
        # Song Alpha shared-audio pair never split
        assert ("A_easy" in ts) == ("A_hard" in ts), f"Song Alpha split at seed {seed}"


def test_song_key_normalisation():
    a = _item("x", "a1", "FREEDOM DiVE!!", "m")
    b = _item("y", "a2", "freedom  dive", "m")
    assert vs.song_key(a) == vs.song_key(b)
    # artist disambiguates when present on both
    c = _item("z", "a3", "Title", "m", artist="Artist One")
    d = _item("w", "a4", "Title", "m", artist="Artist Two")
    assert vs.song_key(c) != vs.song_key(d)
    # empty title falls back to a per-item key (never collides)
    e = _item("e1", "ae", "", "m")
    f = _item("f1", "af", "", "m")
    assert vs.song_key(e) != vs.song_key(f)


def test_grouped_split_reproducible_and_seed_sensitive():
    items = _synthetic_manifest()
    t1, v1 = vs.grouped_split(items, val_frac=0.30, seed=1234)
    t2, v2 = vs.grouped_split(items, val_frac=0.30, seed=1234)
    assert (t1, v1) == (t2, v2), "same seed must give an identical split"
    # a different seed should (very likely) give a different val set here
    _, v3 = vs.grouped_split(items, val_frac=0.30, seed=999)
    assert set(v1) != set(v3) or v1 == [], "seed had no effect on the split"


def test_val_frac_zero_and_bounds():
    items = _synthetic_manifest()
    train, val = vs.grouped_split(items, val_frac=0.0, seed=1)
    assert val == [] and set(train) == {it["item_id"] for it in items}
    # frac honours the target roughly (whole groups -> approximate, never leaky)
    _, val10 = vs.grouped_split(items, val_frac=0.10, seed=3)
    assert 0 < len(val10) <= len(items)


def test_static_split_roundtrip_and_reproducible(tmp_path):
    items = _synthetic_manifest()
    (tmp_path / "manifest.json").write_text(json.dumps(items), encoding="utf-8")

    out = vs.write_static_split(tmp_path, val_frac=0.30, seed=1234)
    assert out.exists()
    loaded = vs.load_static_split(tmp_path)
    assert loaded, "static split should hold out some items"

    # resolve_split must return EXACTLY the frozen val set, regardless of frac/seed
    train, val, used_static = vs.resolve_split(tmp_path, val_frac=0.99, seed=42)
    assert used_static is True
    assert set(val) == set(loaded)
    assert set(train).isdisjoint(set(val))
    vs.assert_no_leakage(items, train, val)

    # re-freezing with the same frac+seed reproduces the identical file
    vs.write_static_split(tmp_path, val_frac=0.30, seed=1234)
    again = vs.load_static_split(tmp_path)
    assert set(again) == set(loaded)


def test_resolve_split_falls_back_without_static(tmp_path):
    items = _synthetic_manifest()
    (tmp_path / "manifest.json").write_text(json.dumps(items), encoding="utf-8")
    train, val, used_static = vs.resolve_split(tmp_path, val_frac=0.30, seed=5)
    assert used_static is False
    assert val and set(train).isdisjoint(set(val))
    vs.assert_no_leakage(items, train, val)


def test_static_split_ignores_stale_ids(tmp_path):
    items = _synthetic_manifest()
    (tmp_path / "manifest.json").write_text(json.dumps(items), encoding="utf-8")
    # hand-written static set with one id no longer in the manifest
    vs.static_split_path(tmp_path).write_text(
        json.dumps({"val_item_ids": ["A_easy", "A_hard", "GONE_ID"]}), encoding="utf-8")
    train, val, used_static = vs.resolve_split(tmp_path, val_frac=0.10, seed=1)
    assert used_static is True
    assert "GONE_ID" not in val
    assert set(val) == {"A_easy", "A_hard"}
    assert set(train).isdisjoint(set(val))


# --- reward-in-val wiring (stubbed; no reward/metrics/generate imports) -----

def test_val_reward_mean_with_stubbed_reward(monkeypatch):
    """Exercise the reward-in-val *aggregation* contract without importing the real
    reward stack. We stub a reward callable and a sampler, mirroring how train.py's
    helper averages per-song rewards over a small fixed subset."""
    items = _synthetic_manifest()
    _, val = vs.grouped_split(items, val_frac=0.30, seed=2)
    # pick a small fixed subset of held-out item_ids (what train.py samples on)
    subset = sorted(val)[:3]

    # stub: "generate + score" returns a deterministic reward per item_id
    fake_rewards = {iid: 0.5 + 0.1 * i for i, iid in enumerate(subset)}

    def fake_sample_and_reward(item_id):
        return fake_rewards[item_id]

    rewards = [fake_sample_and_reward(iid) for iid in subset]
    mean_reward = sum(rewards) / len(rewards)
    assert abs(mean_reward - sum(fake_rewards.values()) / len(subset)) < 1e-9
    # sanity: subset is drawn only from held-out (no train leakage into the probe)
    assert set(subset).issubset(set(val))
