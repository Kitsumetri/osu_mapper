from tqdm import tqdm
import logging
from pathlib import Path

from src.parsing.osu_objects import OsuBeatmap
from src.utils.logger import setup_logger
from src.utils.loader import collect_osu_files


def main() -> None:
    setup_logger(level=logging.INFO, stdout_log=True, file_log=False)
    a = []
    for p in tqdm(collect_osu_files(r'C:\osu!\Songs', limit=3)):
        print(OsuBeatmap(p).metadata)


if __name__ == '__main__':
    main()
