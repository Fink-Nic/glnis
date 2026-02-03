# src/myproject/cli.py
import argparse
from glnis.scripts.gammaloop_state_test import run_state_test
from glnis.scripts.training_prog import run_training_prog


def main() -> None:
    parser = argparse.ArgumentParser(prog="glnis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    state_test = subparsers.add_parser("stest")
    state_test.add_argument('--settings', '-s', type=str,
                            help="The settings .toml file.")

    training_prog = subparsers.add_parser("tprog")
    training_prog.add_argument('--settings', '-s', type=str,
                               help="The settings .toml file.")
    training_prog.add_argument('--comment', type=str, default='No comment.',
                               help="Add a comment to the output summary file.")
    training_prog.add_argument('--no_output', action='store_true', default=False,
                               help="Enable this flag to not output plot/summary file.")

    args = parser.parse_args()

    if args.command == "stest":
        run_state_test(args.settings)
    elif args.command == "tprog":
        run_training_prog(args.settings, args.comment, args.no_output)
