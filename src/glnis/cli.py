# src/myproject/cli.py

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="glnis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    set_def = subparsers.add_parser("setdef", help="Used to set the default path to the GammaLoop examples directory.")
    set_def.add_argument('--directory', '-d', type=str,
                         help="The path to the GammaLoop examples directory.")

    state_test = subparsers.add_parser(
        "stest", help="Tests a settings file by running a single training step and integrating 10k samples.")
    state_test.add_argument('--file', '-f', type=str,
                            help="The settings .toml file.")
    state_test.add_argument('--graph_properties', '-g', action='store_true', default=False,
                            help="Enable this flag to print the graph properties of the integrand.")

    training_prog = subparsers.add_parser("tprog", help="Run and analyze a MadNIS training program.")
    training_prog.add_argument('--file', '-f', type=str,
                               help="The settings .toml file, or a file containing SamplerCompData.")
    training_prog.add_argument('--comment', '-c', type=str, default='No comment.',
                               help="Add a comment to the output summary file.")
    training_prog.add_argument('--no_output', action='store_true', default=False,
                               help="Enable this flag to not save anything to disk.")
    training_prog.add_argument('--no_plot', action='store_true', default=False,
                               help="Enable this flag to not output the plot/summary file.")

    sampler_comparison = subparsers.add_parser("splcomp", help="Compare the performance of different samplers.")
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
    sampler_comparison.add_argument('--no_export', action='store_true', default=False,
                                    help="Enable this flag to not save the sampler states to output.")
    sampler_comparison.add_argument('--plotting_settings', '-p', type=str, default="",
                                    help="Use to overwrite plotting settings.")

    slice_plots = subparsers.add_parser(
        "splots",
        help="Generate slice plots for the integrator states in a given SamplerCompData.")
    slice_plots.add_argument('--file', '-f', type=str,
                             help="File containing either SamplerCompData or SlicePlotsData.")
    slice_plots.add_argument('--settings_file', '-s', type=str, default="",
                             help="The file containing the slice plot settings.")
    slice_plots.add_argument('--state', '-t', type=str, default="no_state_file",
                             help="The file containing the madnis state data.")
    slice_plots.add_argument('--no_output', action='store_true', default=False,
                             help="Enable this flag to not save anything to disk.")
    slice_plots.add_argument('--no_plot', action='store_true', default=False,
                             help="Enable this flag to not output the plot file.")

    mp_efficiency = subparsers.add_parser(
        "mpe", help="Analyze the multi-core performance scaling for the MadNIS integrator.")
    mp_efficiency.add_argument('--file', '-f', type=str,
                               help="The settings .toml file, or a file containing SamplerCompData.")
    mp_efficiency.add_argument('--no_output', action='store_true', default=False,
                               help="Enable this flag to not save anything to disk.")
    mp_efficiency.add_argument('--no_plot', action='store_true', default=False,
                               help="Enable this flag to not output the plot file.")
    mp_efficiency.add_argument('--use_naive', action='store_true', default=False,
                               help="Enable this flag to use the naive integrator.",)
    mp_efficiency.add_argument('--cpu', action='store_true', default=False,
                               help="Enable this flag to run MadNIS on CPU.")

    hp_comp = subparsers.add_parser(
        "hpcomp", help="Compare the performance of different hyperparameter configurations.")
    hp_comp.add_argument('--file', '-f', type=str,
                         help="File containing either settings .toml or HyperparamCompData.")
    hp_comp.add_argument('--recovery_file', '-r', type=str, default=None,
                         help="File containing a HyperparamCompData to recover from, or extend.")
    hp_comp.add_argument('--no_output', action='store_true', default=False,
                         help="Enable this flag to not save anything to disk.")
    hp_comp.add_argument('--no_plot', action='store_true', default=False,
                         help="Enable this flag to not output the plot file(s).")

    gen_ti_examples = subparsers.add_parser("genti", help="Generate example thermal integrand evaluators.")
    gen_ti_examples.add_argument('--force_rebuild', action='store_true', default=False,
                                 help="Rebuild the evaluators, if they already exist.")

    args = parser.parse_args()

    match args.command:
        case "setdef":
            from glnis.scripts.set_default_gl_path import run_set_default_gl_path
            run_set_default_gl_path(path=args.directory,)
        case "stest":
            from glnis.scripts.settings_test import run_settings_test
            run_settings_test(file=args.file,
                              show_graph_properties=args.graph_properties,)
        case "tprog":
            from glnis.scripts.training_prog import run_training_prog
            run_training_prog(file=args.file,
                              comment=args.comment,
                              no_output=args.no_output,
                              no_plot=args.no_plot,)
        case "splcomp":
            from glnis.scripts.sampler_comparison import run_sampler_comp
            run_sampler_comp(file=args.file,
                             comment=args.comment,
                             no_naive=args.no_naive,
                             no_vegas=args.no_vegas,
                             no_havana=args.no_havana,
                             no_output=args.no_output,
                             no_plot=args.no_plot,
                             export_states=not args.no_export,
                             overwrite_plotting_settings=args.plotting_settings)
        case "splots":
            from glnis.scripts.slice_plots import run_slice_plots
            run_slice_plots(file=args.file,
                            settings_file=args.settings_file,
                            no_output=args.no_output,
                            no_plot=args.no_plot,)
        case "mpe":
            from glnis.scripts.multiprocessing_efficiency import run_multiprocessing_efficiency
            run_multiprocessing_efficiency(file=args.file,
                                           no_output=args.no_output,
                                           no_plot=args.no_plot,
                                           use_naive=args.use_naive,
                                           use_cpu=args.cpu,)
        case "hpcomp":
            from glnis.scripts.hyperparam_comparison import run_hyperparam_comparison
            run_hyperparam_comparison(file=args.file,
                                      recovery_file=args.recovery_file,
                                      no_output=args.no_output,
                                      no_plot=args.no_plot,)
        case "genti":
            from glnis.scripts.generate_thermal_integrand_evaluators import run_generate_thermal_integrand_evaluators
            run_generate_thermal_integrand_evaluators(force_rebuild=args.force_rebuild,)
        case _:
            raise ValueError(f"Unknown command {args.command}")
