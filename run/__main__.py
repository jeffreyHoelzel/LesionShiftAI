import argparse
import importlib
import sys

COMMANDS = {
    "train-ensemble": ("train_ensemble_member_cnn", "main"),
    "train-ensemble-member": ("train_ensemble_member_cnn", "main"),
    "train-baseline": ("train_baseline_cnn", "main"),
    "train-vit": ("train_vit", "main"),
    "smoke-data": ("smoke_data_pipeline", "main")
}


def _resolve_command(command: str):
    module_name, fn_name = COMMANDS[command]
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def main() -> None:
    parser = argparse.ArgumentParser(prog="lesionshiftai.pyz")
    parser.add_argument("command", choices=sorted(COMMANDS.keys()))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()

    # forward remaining args to target script main()
    sys.argv = [ns.command, *ns.args]
    _resolve_command(ns.command)()


if __name__ == "__main__":
    main()
