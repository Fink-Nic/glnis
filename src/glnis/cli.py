# src/myproject/cli.py

def main() -> None:
    import argparse
    from glnis.scripts.gammaloop_state_test import run_state_test
    from glnis.scripts.training_prog import run_training_prog
    from glnis.scripts.sampler_comparison import run_sampler_comp
    from glnis.scripts.slice_plots import run_slice_plots
    from glnis.scripts.multiprocessing_efficiency import run_multiprocessing_efficiency
    from glnis.scripts.hyperparam_comparison import run_hyperparam_comparison

    parser = argparse.ArgumentParser(prog="glnis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    state_test = subparsers.add_parser("stest")
    state_test.add_argument('--file', '-f', type=str,
                            help="The settings .toml file.")

    training_prog = subparsers.add_parser("tprog")
    training_prog.add_argument('--file', '-f', type=str,
                               help="The settings .toml file, or a file containing SamplerCompData.")
    training_prog.add_argument('--comment', '-c', type=str, default='No comment.',
                               help="Add a comment to the output summary file.")
    training_prog.add_argument('--no_output', action='store_true', default=False,
                               help="Enable this flag to not save anything to disk.")
    training_prog.add_argument('--no_plot', action='store_true', default=False,
                               help="Enable this flag to not output the plot/summary file.")

    sampler_comparison = subparsers.add_parser("splcomp")
    sampler_comparison.add_argument('--file', '-f', type=str,
                                    help="The settings .toml file, or a file containing SamplerCompData.")
    sampler_comparison.add_argument('--comment', '-c', type=str, default='No comment.',
                                    help="Add a comment to the output summary file.")
    sampler_comparison.add_argument('--no_naive', action='store_true', default=False,
                                    help="Enable this flag to not include the Naive sampler.")
    sampler_comparison.add_argument('--no_vegas', action='store_true', default=False,
                                    help="Enable this flag to not include the Vegas sampler.")
    sampler_comparison.add_argument('--no_havana', action='store_true', default=False,
                                    help="Enable this flag to not include the Havana sampler.")
    sampler_comparison.add_argument('--no_output', action='store_true', default=False,
                                    help="Enable this flag to not save anything to disk.")
    sampler_comparison.add_argument('--no_plot', action='store_true', default=False,
                                    help="Enable this flag to not output the plot file.")

    slice_plots = subparsers.add_parser("splots")
    slice_plots.add_argument('--file', '-f', type=str,
                             help="File containing either SamplerCompData or SlicePlotsData.")
    slice_plots.add_argument('--settings_file', '-s', type=str, default="",
                             help="The file containing the slice plot settings.")
    slice_plots.add_argument('--state', '-t', type=str, default="no_state_file",
                             help="The file containing the madnis state data.")
    slice_plots.add_argument('--comment', '-c', type=str, default='No comment.',
                             help="Add a comment to the output summary file.")
    slice_plots.add_argument('--no_output', action='store_true', default=False,
                             help="Enable this flag to not save anything to disk.")
    slice_plots.add_argument('--no_plot', action='store_true', default=False,
                             help="Enable this flag to not output the plot file.")

    mp_efficiency = subparsers.add_parser("mpe")
    mp_efficiency.add_argument('--file', '-f', type=str,
                               help="File containing either SamplerCompData or TrainingProgData.")
    mp_efficiency.add_argument('--state', '-t', type=str, default="no_state_file",
                               help="The file containing the madnis state data.")
    mp_efficiency.add_argument('--comment', '-c', type=str, default='No comment.',
                               help="Add a comment to the output summary file.")
    mp_efficiency.add_argument('--no_output', action='store_true', default=False,
                               help="Enable this flag to not save anything to disk.")
    mp_efficiency.add_argument('--no_plot', action='store_true', default=False,
                               help="Enable this flag to not output the plot file.")

    hp_comp = subparsers.add_parser("hpcomp")
    hp_comp.add_argument('--file', '-f', type=str,
                         help="File containing either settings .toml or HyperparamCompData.")
    hp_comp.add_argument('--recovery_file', '-r', type=str, default=None,
                         help="File containing a HyperparamCompData to recover from.")
    hp_comp.add_argument('--comment', '-c', type=str, default='No comment.',
                         help="Add a comment to the output summary file.")
    hp_comp.add_argument('--no_output', action='store_true', default=False,
                         help="Enable this flag to not save anything to disk.")
    hp_comp.add_argument('--no_plot', action='store_true', default=False,
                         help="Enable this flag to not output the plot file(s).")

    args = parser.parse_args()

    match args.command:
        case "stest":
            run_state_test(file=args.file,)
        case "tprog":
            run_training_prog(file=args.file,
                              comment=args.comment,
                              no_output=args.no_output,
                              no_plot=args.no_plot,)
        case "splcomp":
            run_sampler_comp(file=args.file,
                             comment=args.comment,
                             no_naive=args.no_naive,
                             no_vegas=args.no_vegas,
                             no_havana=args.no_havana,
                             no_output=args.no_output,
                             no_plot=args.no_plot,)
        case "splots":
            run_slice_plots(file=args.file,
                            settings_file=args.settings_file,
                            comment=args.comment,
                            no_output=args.no_output,
                            no_plot=args.no_plot,)
        case "mpe":
            run_multiprocessing_efficiency(file=args.file,
                                           comment=args.comment,
                                           no_output=args.no_output,
                                           no_plot=args.no_plot,)
        case "hpcomp":
            run_hyperparam_comparison(file=args.file,
                                      recovery_file=args.recovery_file,
                                      comment=args.comment,
                                      no_output=args.no_output,
                                      no_plot=args.no_plot,)
        case _:
            raise ValueError(f"Unknown command {args.command}")
