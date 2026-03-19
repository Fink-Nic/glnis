# src/myproject/cli.py

def main() -> None:
    import argparse
    from glnis.scripts.gammaloop_state_test import run_state_test
    from glnis.scripts.training_prog import run_training_prog
    from glnis.scripts.sampler_comparison import run_sampler_comp
    from glnis.scripts.slice_plots import run_slice_plots
    from glnis.scripts.multiprocessing_efficiency import run_multiprocessing_efficiency

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

    slice_plots = subparsers.add_parser("splots")
    slice_plots.add_argument('--settings', '-s', type=str,
                             help="The settings .toml file.")
    slice_plots.add_argument('--state', '-t', type=str, default="no_state_file",
                             help="The file containing the madnis state data.")
    slice_plots.add_argument('--comment', '-c', type=str, default='No comment.',
                             help="Add a comment to the output summary file.")
    slice_plots.add_argument('--no_output', action='store_true', default=False,
                             help="Enable this flag to not output plot/summary file.")
    slice_plots.add_argument('--only_plot', action='store_true', default=False,
                             help="Enable this flag to only output the plot/summary file from a previous run.")
    slice_plots.add_argument('--no_plot', action='store_true', default=False,
                             help="Enable this flag to not output the plot file.")

    mp_efficiency = subparsers.add_parser("mpe")
    mp_efficiency.add_argument('--settings', '-s', type=str,
                               help="The settings .toml file.")
    mp_efficiency.add_argument('--state', '-t', type=str, default="no_state_file",
                               help="The file containing the madnis state data.")
    mp_efficiency.add_argument('--comment', '-c', type=str, default='No comment.',
                               help="Add a comment to the output summary file.")
    mp_efficiency.add_argument('--no_output', action='store_true', default=False,
                               help="Enable this flag to not output plot/summary file.")
    mp_efficiency.add_argument('--no_plot', action='store_true', default=False,
                               help="Enable this flag to not output the plot file.")
    mp_efficiency.add_argument('--only_plot', action='store_true', default=False,
                               help="Enable this flag to only output the plot/summary file from a previous run.")

    args = parser.parse_args()

    match args.command:
        case "stest":
            run_state_test(settings_file=args.settings,)
        case "tprog":
            run_training_prog(file=args.settings,
                              comment=args.comment,
                              no_output=args.no_output,
                              only_plot=args.only_plot,
                              no_plot=args.no_plot,)
        case "splcomp":
            run_sampler_comp(file=args.settings,
                             comment=args.comment,
                             no_naive=args.no_naive,
                             no_vegas=args.no_vegas,
                             no_havana=args.no_havana,
                             no_output=args.no_output,
                             only_plot=args.only_plot,
                             no_plot=args.no_plot,)
        case "splots":
            run_slice_plots(settings_file=args.settings,
                            state_file=args.state,
                            comment=args.comment,
                            no_output=args.no_output,
                            only_plot=args.only_plot,
                            no_plot=args.no_plot,)
        case "mpe":
            run_multiprocessing_efficiency(settings_file=args.settings,
                                           state_file=args.state,
                                           comment=args.comment,
                                           no_output=args.no_output,
                                           only_plot=args.only_plot,
                                           no_plot=args.no_plot,)
        case _:
            raise ValueError(f"Unknown command {args.command}")
