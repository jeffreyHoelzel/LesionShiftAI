import argparse
import sys
from smoke_data_pipeline import main as smoke_data_main
from train_baseline_cnn import main as train_baseline_main

COMMANDS = {
    "train-baseline": train_baseline_main,
    "smoke-data": smoke_data_main,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="lesionshiftai.pyz")
    parser.add_argument("command", choices=sorted(COMMANDS.keys()))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()

    # forward remaining args to target script main()
    sys.argv = [ns.command, *ns.args]
    COMMANDS[ns.command]()


if __name__ == "__main__":
    main()
