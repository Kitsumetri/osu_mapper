import os
from pathlib import Path


def collect_osu_files(root_dir: str | Path, limit: int | None = None):
    osu_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".osu"):
                osu_files.append(os.path.join(dirpath, filename))
                if limit is not None and len(osu_files) >= limit:
                    return osu_files

    return osu_files
