"""osu_mapper CLI dispatcher.

Thin wrapper around the pipeline stages so the whole project is driveable from
one entrypoint. Each stage also has a module entrypoint (``python -m src.*``).

  python main.py preprocess --songs "C:/osu!/Songs" --out data/processed --limit 600
  python main.py train      --data data/processed --epochs 240 --batch 8
  python main.py generate   --audio song.mp3 --ckpt checkpoints/model_last.pt --out out.osu
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(
        prog="osu_mapper", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preprocess", add_help=False)
    sub.add_parser("train", add_help=False)
    sub.add_parser("generate", add_help=False)

    args, rest = ap.parse_known_args()
    # hand the remaining args to the chosen stage's own argparse
    sys.argv = [f"src.{args.cmd}"] + rest
    if args.cmd == "preprocess":
        from src.data.preprocess import main as run
    elif args.cmd == "train":
        from src.train import main as run
    elif args.cmd == "generate":
        from src.generate import main as run
    run()


if __name__ == "__main__":
    main()
