"""__main__.py

Main program entry-point for all model training, validation, and testing. 
Supports each as a CLI function. When built, this becomes the entry-point for the 
`dist/lesionshiftai.pyz` script.
"""
import argparse
import importlib
import sys

COMMANDS = {
    "train-ensemble": ("train_ensemble_member_cnn", "main"),
    "train-ensemble-member": ("train_ensemble_member_cnn", "main"),
    "train-baseline": ("train_baseline_cnn", "main"),
    "train-vit": ("train_vit", "main")
}


def _resolve_command(command: str):
    """
    Ensures command exists in look-up table then passes back main 
    entry-point for training step.
    """
    module_name, fn_name = COMMANDS[command]
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def main() -> None:
    """Main function that executes selected training module."""
    parser = argparse.ArgumentParser(prog="lesionshiftai.pyz")
    parser.add_argument("command", choices=sorted(COMMANDS.keys()))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()

    # forward remaining args to target script main()
    sys.argv = [ns.command, *ns.args]
    _resolve_command(ns.command)()


if __name__ == "__main__":
    main()
