import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import colorlog

ROOT_DIR: Path = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent


def setup_logger(level: Optional[int] = logging.NOTSET,
                 stdout_log: Optional[bool] = True,
                 file_log: Optional[bool] = True) -> None:
    if not (stdout_log or file_log):
        exit(">>> stdout and file logs are False")

    handlers = []
    log_filename = Path()

    if file_log:
        log_filename: Path = Path(
            ROOT_DIR / f"logs/logs_{datetime.now():%S-%m-%d-%Y}.log"
        ).resolve()

        os.makedirs(log_filename.parent, exist_ok=True)
        handlers.append(logging.FileHandler(log_filename))

    if stdout_log:
        color_formatter = colorlog.ColoredFormatter(
            '%(log_color)s>>> %(module)s:%(lineno)d - %(levelname)s - %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(color_formatter)
        handlers.append(stream_handler)

    logging.basicConfig(
        level=level,
        format='>>> %(module)s:%(lineno)d - %(levelname)s - %(message)s',
        handlers=handlers
    )

    if file_log:
        logging.info(f"Log ({file_log=}, {stdout_log=}) file was created at {log_filename}")
    else:
        logging.warning(f"Log file wasn't created due to {file_log=}")