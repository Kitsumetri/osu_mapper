"""Leakage-free, group-aware train/val split over a processed-dataset manifest.

The processed dataset stores ONE mel per audio file (``mels/<audio_id>.npy``) that
is SHARED by every difficulty of that song (``dataset.py`` loads it per item). The
old split (``torch.randperm`` over manifest items in ``train.py``) therefore leaks:
a song's difficulties scatter across train and val, so the *same audio* the model
trained on appears in validation -> ``val_loss`` is optimistically low and is not a
real held-out signal.

Worse, the SAME song (same title/artist) is frequently mapped by DIFFERENT mappers,
sometimes with a slightly different audio file (length differs a second or two, or a
different encoding), so the audio bytes -> a different ``audio_id``. Deduping by
``audio_id`` alone is therefore insufficient: a song can still leak across the split
via a second mapper's copy.

The fix here groups items by a normalised **song-identity key** (title, plus artist
when the manifest has it) AND keeps any shared ``audio_id`` on the same side, then
holds out *whole groups*. No song-identity key and no ``audio_id`` is shared between
train and val.

This module is intentionally **dependency-light** — it imports only the stdlib so it
can be tested in total isolation (no torch / dataset / GPU / reward). ``train.py``
calls :func:`grouped_split` (or loads a frozen :func:`load_static_split`).

CLI — freeze a reproducible static val set (so it's identical across configs/runs):

    uv run python -m src.data.val_split --data data/processed/<tag> --frac 0.10 --seed 1234
    # -> writes data/processed/<tag>/val_split.json  (the held-out item_ids)
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# default static-split filename inside a processed-dataset dir
VAL_SPLIT_FILE = "val_split.json"

# punctuation / symbols collapsed to a single space when normalising a title/artist
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_WS = re.compile(r"\s+")


def _norm_text(s: str) -> str:
    """Lowercase, strip, drop punctuation/symbols, collapse whitespace.

    "Through the Fire and Flames!!" and "through  the fire and flames" both
    normalise to "through the fire and flames", so the same song under cosmetic
    title differences (extra spaces, punctuation, case) maps to one key.
    """
    if not s:
        return ""
    s = _NON_WORD.sub(" ", str(s).lower().strip())
    return _WS.sub(" ", s).strip()


def song_key(item: dict) -> str:
    """Normalised song-identity key for a manifest item.

    Built from the *song*, NOT the mapper, so a song mapped by different creators
    (the duplicate-audio-across-mappers case) collapses to one key and cannot leak
    across the split. Uses ``artist`` when the manifest has it (older manifests only
    store ``title`` — ``preprocess.py`` doesn't persist artist — so we degrade to
    title-only, which is still correct: the worst case is two genuinely-different
    songs that share a title getting grouped together, which only makes the holdout
    slightly more conservative, never leakier).
    """
    title = _norm_text(item.get("title", ""))
    artist = _norm_text(item.get("artist", ""))     # usually absent -> ""
    key = f"{artist}\x1f{title}" if artist else title
    return key or f"__item__{item.get('item_id', '')}"  # never empty -> per-item


def _union_find_groups(items: list[dict]) -> dict[str, list[int]]:
    """Cluster item indices that must stay on the same side of the split.

    Two items are linked if they share a song-identity key OR share an ``audio_id``.
    Linking by BOTH and taking the transitive closure means: a song mapped by two
    creators (linked by song_key) and a second audio file of that song (linked into
    the same component the moment any item references either id) all land together.
    Returns ``{component_root_label: [item_index, ...]}``.

    Implemented as a tiny union-find over two label namespaces ("k:" song keys and
    "a:" audio ids); each item unions its song-key node with its audio-id node, so a
    component spans every key/audio reachable through any shared item.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:        # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # node per item too, so an item with neither a usable key nor audio_id still
    # forms its own component (it can't be silently dropped).
    for i, it in enumerate(items):
        knode = "k:" + song_key(it)
        inode = "i:" + str(i)
        find(knode)
        find(inode)
        union(inode, knode)
        aid = it.get("audio_id")
        if aid:
            union(knode, "a:" + str(aid))

    groups: dict[str, list[int]] = {}
    for i in range(len(items)):
        root = find("i:" + str(i))
        groups.setdefault(root, []).append(i)
    return groups


def grouped_split(items: list[dict], val_frac: float = 0.10,
                  seed: int = 1234) -> tuple[list[str], list[str]]:
    """Leakage-free group-aware split.

    Groups items by song-identity (title[+artist]) unioned with shared ``audio_id``,
    shuffles the *groups* deterministically by ``seed``, and assigns whole groups to
    val until ~``val_frac`` of items are held out. Guarantees: **no song-identity key
    and no ``audio_id`` is shared between the returned train and val item-id lists.**

    Returns ``(train_item_ids, val_item_ids)``. ``val_frac <= 0`` -> ``(all, [])``.
    The order of the returned ids follows the manifest order (stable) so downstream
    Subset indices are deterministic.
    """
    item_ids = [it["item_id"] for it in items]
    if val_frac <= 0 or len(items) == 0:
        return list(item_ids), []

    groups = _union_find_groups(items)
    group_list = sorted(groups.values(), key=lambda g: g[0])  # deterministic base order
    rng = random.Random(seed)
    rng.shuffle(group_list)

    n_total = len(items)
    n_val_target = round(n_total * val_frac)
    val_set: set[int] = set()
    for grp in group_list:
        if len(val_set) >= n_val_target:
            break
        val_set.update(grp)

    val_ids = [item_ids[i] for i in range(n_total) if i in val_set]
    train_ids = [item_ids[i] for i in range(n_total) if i not in val_set]
    return train_ids, val_ids


def assert_no_leakage(items: list[dict], train_ids, val_ids) -> None:
    """Raise ``AssertionError`` if any song-key or audio_id is shared across sides.

    Cheap invariant check usable as a guard in ``train.py`` and asserted by tests.
    """
    by_id = {it["item_id"]: it for it in items}
    val_idset = set(val_ids)

    def keys_for(ids):
        sk, aid = set(), set()
        for iid in ids:
            it = by_id.get(iid)
            if it is None:
                continue
            sk.add(song_key(it))
            if it.get("audio_id"):
                aid.add(it["audio_id"])
        return sk, aid

    if not val_idset:
        return
    tk, ta = keys_for(train_ids)
    vk, va = keys_for(val_ids)
    leaked_keys = tk & vk
    leaked_audio = ta & va
    assert not leaked_keys, f"song-key leakage across split: {sorted(leaked_keys)[:5]}"
    assert not leaked_audio, f"audio_id leakage across split: {sorted(leaked_audio)[:5]}"


# --- static (frozen) split -------------------------------------------------

def load_manifest(processed_dir: str | Path) -> list[dict]:
    """Read ``<dir>/manifest.json`` (the list of item dicts)."""
    path = Path(processed_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"no manifest.json in {processed_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def static_split_path(processed_dir: str | Path) -> Path:
    return Path(processed_dir) / VAL_SPLIT_FILE


def load_static_split(processed_dir: str | Path) -> list[str] | None:
    """Return the frozen held-out item_ids from ``<dir>/val_split.json``, or None.

    The stored value can be either a bare list of item_ids or a dict with a
    ``"val_item_ids"`` key (the format :func:`write_static_split` emits); both are
    accepted so a hand-written file works too.
    """
    p = static_split_path(processed_dir)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("val_item_ids", []))
    return list(data)


def write_static_split(processed_dir: str | Path, val_frac: float = 0.10,
                       seed: int = 1234) -> Path:
    """Compute the grouped split and freeze the val item_ids to ``val_split.json``.

    Reproducible: same manifest + frac + seed -> identical file. Records the frac /
    seed / counts alongside the ids for provenance. The static set means the val set
    is identical across every config/run on this dataset (the user's requirement).
    Returns the written path.
    """
    items = load_manifest(processed_dir)
    train_ids, val_ids = grouped_split(items, val_frac=val_frac, seed=seed)
    assert_no_leakage(items, train_ids, val_ids)
    out = static_split_path(processed_dir)
    payload = {
        "val_frac": val_frac,
        "seed": seed,
        "n_total": len(items),
        "n_val": len(val_ids),
        "n_train": len(train_ids),
        "val_item_ids": val_ids,
    }
    out.write_text(json.dumps(payload, indent=0), encoding="utf-8")
    return out


def resolve_split(processed_dir: str | Path, val_frac: float = 0.10,
                  seed: int = 1234) -> tuple[list[str], list[str], bool]:
    """Split used by training: a frozen static set if present, else a fresh grouped
    split with the fixed ``seed``.

    Returns ``(train_item_ids, val_item_ids, used_static)``. When a static
    ``val_split.json`` exists, its held-out ids define val (intersected with the
    current manifest so a stale id is ignored) and everything else is train; the
    no-leakage invariant is still asserted.
    """
    items = load_manifest(processed_dir)
    static = load_static_split(processed_dir)
    if static is not None:
        present = {it["item_id"] for it in items}
        val_ids = [iid for iid in static if iid in present]
        val_set = set(val_ids)
        train_ids = [it["item_id"] for it in items if it["item_id"] not in val_set]
        assert_no_leakage(items, train_ids, val_ids)
        return train_ids, val_ids, True
    train_ids, val_ids = grouped_split(items, val_frac=val_frac, seed=seed)
    assert_no_leakage(items, train_ids, val_ids)
    return train_ids, val_ids, False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Freeze a reproducible, leakage-free val split for a processed "
                    "dataset (writes <data>/val_split.json with the held-out item_ids).")
    ap.add_argument("--data", required=True, help="processed dataset dir (has manifest.json)")
    ap.add_argument("--frac", type=float, default=0.10,
                    help="fraction of items to hold out for validation (default 0.10)")
    ap.add_argument("--seed", type=int, default=1234, help="shuffle seed (default 1234)")
    args = ap.parse_args()
    out = write_static_split(args.data, val_frac=args.frac, seed=args.seed)
    data = json.loads(out.read_text(encoding="utf-8"))
    print(f"wrote {out}")
    print(f"  held out {data['n_val']}/{data['n_total']} items "
          f"(frac {args.frac}, seed {args.seed}); train {data['n_train']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
