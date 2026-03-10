# src/myproject/cli.py

def main() -> None:
    import argparse
    from glnis.scripts.gammaloop_state_test import run_state_test
    from glnis.scripts.training_prog import run_training_prog
    from glnis.scripts.sampler_comparison import run_sampler_comp

    parser = argparse.ArgumentParser(prog="glnis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    state_test = subparsers.add_parser("stest")
    state_test.add_argument('--settings', '-s', type=str,
                            help="The settings .toml file.")

    training_prog = subparsers.add_parser("tprog")
    training_prog.add_argument('--settings', '-s', type=str,
                               help="The settings .toml file.")
    training_prog.add_argument('--comment', '-c', type=str, default='No comment.',
                               help="Add a comment to the output summary file.")
    training_prog.add_argument('--no_output', action='store_true', default=False,
                               help="Enable this flag to not output plot/summary file.")
    training_prog.add_argument('--only_plot', action='store_true', default=False,
                               help="Enable this flag to only output the plot/summary file from a previous run.")
    training_prog.add_argument('--no_plot', action='store_true', default=False,
                               help="Enable this flag to not output the plot/summary file.")

    sampler_comparison = subparsers.add_parser("splcomp")
    sampler_comparison.add_argument('--settings', '-s', type=str,
                                    help="The settings .toml file.")
    sampler_comparison.add_argument('--comment', '-c', type=str, default='No comment.',
                                    help="Add a comment to the output summary file.")
    sampler_comparison.add_argument('--no_naive', action='store_true', default=False,
                                    help="Enable this flag to not include the Naive sampler.")
    sampler_comparison.add_argument('--no_vegas', action='store_true', default=False,
                                    help="Enable this flag to not include the Vegas sampler.")
    sampler_comparison.add_argument('--no_havana', action='store_true', default=False,
                                    help="Enable this flag to not include the Havana sampler.")
    sampler_comparison.add_argument('--no_output', action='store_true', default=False,
                                    help="Enable this flag to not output plot/summary file.")
    sampler_comparison.add_argument('--no_plot', action='store_true', default=False,
                                    help="Enable this flag to not output the plot file.")
    sampler_comparison.add_argument('--only_plot', action='store_true', default=False,
                                    help="Enable this flag to only output the plot/summary file from a previous run.")

    args = parser.parse_args()

    if args.command == "stest":
        run_state_test(settings_file=args.settings,)
    elif args.command == "tprog":
        run_training_prog(file=args.settings,
                          comment=args.comment,
                          no_output=args.no_output,
                          only_plot=args.only_plot,
                          no_plot=args.no_plot,)
    elif args.command == "splcomp":
        run_sampler_comp(file=args.settings,
                         comment=args.comment,
                         no_naive=args.no_naive,
                         no_vegas=args.no_vegas,
                         no_havana=args.no_havana,
                         no_output=args.no_output,
                         only_plot=args.only_plot,
                         no_plot=args.no_plot,)
