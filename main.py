"""osu_mapper CLI — one entrypoint for the whole pipeline.

  python main.py preprocess --songs "C:/osu!/Songs" --out data/processed/ranked --gold --workers 10
  python main.py train      --data data/processed/ranked --tag mymodel --base 160 --epochs 60
  python main.py infer    --audio song.mp3 --reference ref.osu --sr 5 6   # generate + package
  python main.py generate --audio song.mp3 --ckpt runs/<id>/ckpt/best.pt --out out.osu

Each stage also has a module entrypoint (python -m src.train, python -m src.run_inference, ...).
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(
        prog="main.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    # add_help=False so a stage's own --help passes through to it (not handled here)
    for name in ("preprocess", "train", "infer", "generate"):
        sub.add_parser(name, add_help=False)

    args, rest = ap.parse_known_args()
    sys.argv = [f"main.py {args.cmd}"] + rest   # hand the rest to the stage's own argparse
    if args.cmd == "preprocess":
        from src.data.preprocess import main as run
    elif args.cmd == "train":
        from src.train import main as run
    elif args.cmd == "infer":
        from src.run_inference import main as run
    elif args.cmd == "generate":
        from src.generate import main as run
    raise SystemExit(run() or 0)


if __name__ == "__main__":
    main()
