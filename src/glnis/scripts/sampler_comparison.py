# type: ignore
import numpy as np
import pickle
from numpy.typing import NDArray
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from traceback import print_exc
from madnis.integrator import TrainingStatus as MadnisTrainingStatus

from glnis.utils.helpers import shell_print, verify_path
from glnis.core.accumulator import (
    GraphProperties,
    IntegrationResult,
    DefaultAccumulator,
    TrainingAccumulator
)

MADNIS_KEY = "MadNIS"
NAIVE_KEY = "Naive"
VEGAS_KEY = "Vegas"
HAVANA_KEY = "Havana"


@dataclass
class TrainingProgress:
    losses: List[float] | None = field(default_factory=list)
    rsds: List[float] = field(default_factory=list)
    abs_rsds: List[float] = field(default_factory=list)
    tvars: List[float] = field(default_factory=list)
    abs_tvars: List[float] = field(default_factory=list)
    steps_losses: List[int] = field(default_factory=list)
    nspl_losses: List[int] = field(default_factory=list)
    steps_snapshot: List[int] = field(default_factory=list)
    nspl_snapshot: List[int] = field(default_factory=list)
    means: List[float] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)
    steps_discrete: List[int] = field(default_factory=list)
    nspl_discrete: List[int] = field(default_factory=list)
    all_channels: NDArray = field(default_factory=lambda: np.array([]))
    discrete_probs: List[NDArray] = field(default_factory=list)


class SamplerCompData:
    def __init__(self,
                 integrator_identifiers: List[str],
                 graph_properties: GraphProperties,
                 target: IntegrationResult,
                 settings: Dict[str, Any] = None,
                 madnis_kwargs: Dict[str, Any] = None,
                 madnis_info: Dict[str, Any] = None,
                 integrand_kwargs: Dict[str, Any] = None,
                 param_kwargs: Dict[str, Any] = None,) -> None:
        self.result: Dict[str, IntegrationResult] = dict()
        self.observables: Dict[str, Dict[str, Any]] = dict()
        self.training_progress: Dict[str, TrainingProgress] = dict()
        for name in integrator_identifiers:
            self.result[name] = IntegrationResult()
            self.observables[name] = dict()
            if not name == NAIVE_KEY:
                self.training_progress[name] = TrainingProgress()
        self.graph_properties = graph_properties
        self.target = target
        self.settings: Dict[str, Any] = settings or dict()
        self.madnis_info: Dict[str, Any] = madnis_info or dict()
        self.madnis_kwargs: Dict[str, Any] = madnis_kwargs or dict()
        self.integrand_kwargs: Dict[str, Any] = integrand_kwargs or dict()
        self.param_kwargs: Dict[str, Any] = param_kwargs or dict()
        self.integrator_states: Dict[str, Any] = dict()
        for name in integrator_identifiers:
            self.integrator_states[name] = None

    def __setstate__(self, state):
        """Custom unpickling logic to handle changes in the class definition over time."""
        self.__dict__.update(state)

        # Provide defaults for fields that didn't exist when the file was saved
        if 'madnis_info' not in state:
            self.madnis_info = dict()


def run_sampler_comp(
    file: str | Dict,
    comment: str = "",
    no_naive: bool = False,
    no_vegas: bool = False,
    no_havana: bool = False,
    no_output: bool = False,
    no_plot: bool = False,
    export_states: bool = True,
    subroutine: str = "sampler_comp",
    overwrite_plotting_settings: str = "",
) -> SamplerCompData | None:

    import signal
    import gc
    from time import time

    from glnis.core.integrator import (
        Integrator,
        NaiveIntegrator,
        VegasIntegrator,
        HavanaIntegrator,
        MadnisIntegrator,
    )
    from glnis.core.parser import SettingsParser

    signal.signal(signal.SIGINT, signal.default_int_handler)
    integrators: dict[str, Integrator] = dict()
    cleanup_done = False

    def end_all_integrators() -> None:
        nonlocal cleanup_done
        if cleanup_done:
            return
        for integrator in integrators.values():
            integrator.free()
        # Force prompt cleanup of cycles (e.g. mp queues/process wrappers)
        # in repeated run_sampler_comp calls.
        gc.collect()
        cleanup_done = True

    try:
        shell_print(f"Working on {"dictionary" if isinstance(file, dict) else file}")
        try:
            Settings = SettingsParser(file)
            SData = None
        except:
            with verify_path(file).open('rb') as f:
                SData = pickle.load(f)
            if not isinstance(SData, SamplerCompData):
                raise ValueError(
                    f"Expected a SamplerCompData object in the file, but got {type(SData)}")
            Settings = SettingsParser(SData.settings)
        settings_for_plotting = Settings.settings
        if overwrite_plotting_settings:
            try:
                settings_for_plotting = SettingsParser(overwrite_plotting_settings).settings
            except:
                shell_print(
                    f"Could not parse plotting settings from {overwrite_plotting_settings}, using original settings")
        scripts_for_plotting: Dict[str, Dict[str, Any]] = settings_for_plotting.get("scripts", dict())
        plotting_settings = scripts_for_plotting.get(subroutine, dict()).get("plotting", dict())
        if len(plotting_settings) == 0:
            plotting_settings = scripts_for_plotting.get("sampler_comp", dict()).get("plotting", dict())

        if isinstance(SData, SamplerCompData):
            plot_sampler_comp(file, SData, comment, plotting_settings)
            return SData

        # Training parameters
        scripts: Dict[str, Any] = Settings.settings.get("scripts", dict())
        params: Dict[str, Any] = scripts.get(subroutine, dict())
        if len(params) == 0:
            params = scripts.get("sampler_comp", dict())
        n_training_steps = params.get("n_training_steps", 1000)
        n_log = params.get("n_log", 10)
        n_plot_snapshot = params.get("n_plot_snapshot", 100)
        n_plot_loss = params.get("n_plot_loss", 2)
        plot_disc = params.get("plot_disc", True)
        n_plot_disc = params.get("n_plot_disc", 10)
        n_samples = params.get("n_samples", 10_000)
        n_samples_after_training = params.get("n_samples_after_training", 100_000)
        nitn = params.get("nitn", 10)  # number of vegas/havana training iterations

        if not no_output:
            PROJECT_ROOT = Path(__file__).parents[3]
            OUTPUT_DIR = "outputs"
            directory = Path(PROJECT_ROOT, OUTPUT_DIR, Settings.settings.get("output_dir", "default"), subroutine)
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                shell_print(f"Created output folder at {directory}")
            shell_print(f"Output will be at {directory}")

        time_last = time()
        Settings.settings["layered_integrator"]["integrator_type"] = "madnis"
        madnis_kwargs = Settings.get_integrator_kwargs()
        integrand_kwargs = Settings.get_integrand_kwargs()
        param_kwargs = Settings.get_parameterisation_kwargs()
        madnis_integrator: MadnisIntegrator = Integrator.from_settings(
            Settings.settings
        )
        integrators[MADNIS_KEY] = madnis_integrator

        if not no_naive:
            Settings.settings["layered_integrator"]["integrator_type"] = "naive"
            integrators[NAIVE_KEY] = NaiveIntegrator(madnis_integrator.integrand, **Settings.get_integrator_kwargs())
        if not no_vegas:
            Settings.settings["layered_integrator"]["integrator_type"] = "vegas"
            integrators[VEGAS_KEY] = VegasIntegrator(madnis_integrator.integrand, **Settings.get_integrator_kwargs())
        if not no_havana:
            Settings.settings["layered_integrator"]["integrator_type"] = "havana"
            integrators[HAVANA_KEY] = HavanaIntegrator(madnis_integrator.integrand, **Settings.get_integrator_kwargs())

        for integrator in integrators.values():
            integrator.display_info()

        # Will hold integration results to write to text file and plot
        Data = SamplerCompData(integrator_identifiers=list(integrators.keys()),
                               graph_properties=madnis_integrator.integrand.graph_properties,
                               target=madnis_integrator.integrand.target,
                               settings=Settings.settings,
                               madnis_kwargs=madnis_kwargs,
                               madnis_info=madnis_integrator.get_info(),
                               integrand_kwargs=integrand_kwargs,
                               param_kwargs=param_kwargs,)

        # Callback for the madnis integrator
        if len(madnis_integrator.integrand.discrete_dims) > int(madnis_integrator.integrand.strat_sgn):
            plot_disc = madnis_integrator._discrete_prod <= 10000 and plot_disc
        else:
            plot_disc = False
        if plot_disc:
            all_channels = np.array(np.meshgrid(
                *[range(dim)
                  for dim in madnis_integrator.discrete_dims
                  [int(madnis_integrator.integrand.strat_sgn):]])).T.reshape(
                -1, len(madnis_integrator.discrete_dims))
            # Discard the ones with zero prior probability
            prior = madnis_integrator.integrand.apply_prior_to_discrete(all_channels)
            all_channels = all_channels[prior > 0]
            for tp in Data.training_progress.values():
                tp.all_channels = all_channels

        def add_integration_snapshot(itg: Integrator, tp: TrainingProgress, print_results: bool = False) -> None:
            acc: TrainingAccumulator = itg.integrate(
                n_samples, progress_report=False, acc_type='training')
            tp.means.append(acc.training_mean)
            tp.errors.append(acc.training_err)
            tp.rsds.append(acc.training_rsd)
            tp.abs_rsds.append(acc.training_abs_rsd)
            tp.tvars.append(acc.training_tvar)
            tp.abs_tvars.append(acc.training_abs_tvar)
            tp.steps_snapshot.append(itg.step)
            tp.nspl_snapshot.append(itg.total_training_samples)
            if print_results:
                shell_print(
                    f"Trained Result after {itg.step} steps | {itg.total_training_samples} total training samples:")
                shell_print(acc.str_report())

        def add_disc_prob_snapshot(
                itg: Integrator,
                tp: TrainingProgress,) -> None:
            tp.steps_discrete.append(itg.step)
            tp.nspl_discrete.append(itg.total_training_samples)
            tp.discrete_probs.append(itg.probe_prob(discrete=tp.all_channels))

        def callback_madnis(status: MadnisIntegrator.TrainingStatus) -> None:
            step = status.step
            if step % n_log == 0:
                shell_print(
                    f"Step {step}: loss={status.madnis_status.loss:.5f}, lr = {status.madnis_status.learning_rate:.2e}"
                )
            if step % n_plot_loss == 0:
                Data.training_progress[MADNIS_KEY].losses.append(status.madnis_status.loss)
                Data.training_progress[MADNIS_KEY].steps_losses.append(step)
                Data.training_progress[MADNIS_KEY].nspl_losses.append(status.total_samples)
            if step % n_plot_snapshot == 0:
                add_integration_snapshot(madnis_integrator, Data.training_progress[MADNIS_KEY], True)
            if step % n_plot_disc == 0 and plot_disc:
                add_disc_prob_snapshot(
                    madnis_integrator,
                    Data.training_progress[MADNIS_KEY],
                )

        def callback(KEY: str, status: Integrator.TrainingStatus) -> None:
            itg = integrators[KEY]
            add_integration_snapshot(itg, Data.training_progress[KEY])
            if plot_disc:
                add_disc_prob_snapshot(itg, Data.training_progress[KEY])

        n_total_training_samples = n_training_steps * madnis_kwargs["batch_size"]
        neval = int(n_total_training_samples / nitn)

        shell_print(
            f"""Initializing the Integrand and Integrators took {
                -time_last + (time_last := time()):.2f}s"""
        )

        shell_print("Taking preliminary snapshots...")
        for identifier, itg in integrators.items():
            if identifier == NAIVE_KEY:
                continue
            add_integration_snapshot(itg, Data.training_progress[identifier], True)
            if plot_disc:
                add_disc_prob_snapshot(itg, Data.training_progress[identifier])

        # Training all the integrators
        if not no_vegas:
            shell_print(f"Training {VEGAS_KEY}:")
            vegas_training_result = integrators[VEGAS_KEY].train(
                nitn, neval, callback=lambda status: callback(VEGAS_KEY, status))
            shell_print(vegas_training_result)
            shell_print(f"Training {VEGAS_KEY} took {-time_last + (time_last := time()): .2f}s")

        if not no_havana:
            shell_print(f"Training {HAVANA_KEY}:")
            havana_training_result = integrators[HAVANA_KEY].train(
                nitn, neval, callback=lambda status: callback(HAVANA_KEY, status))
            shell_print(havana_training_result)
            shell_print(f"Training {HAVANA_KEY} took {-time_last + (time_last := time()): .2f}s")

        shell_print(f"Training {MADNIS_KEY}:")
        madnis_integrator.train(n_training_steps, callback=callback_madnis)
        shell_print(f"Training {MADNIS_KEY} took {-time_last + (time_last := time()): .2f}s")

        for identifier, integrator in integrators.items():
            if export_states:
                Data.integrator_states[identifier] = integrator.export_state()
            shell_print(f"Integrating using {identifier}:")
            acc: DefaultAccumulator = integrator.integrate(n_samples_after_training)
            Data.result[identifier] = acc.statistics.result
            Data.observables[identifier].update(acc.get_observables())

        # IMPORTANT: close the worker functions, or your script will hang
        end_all_integrators()

        if no_output:
            return Data

        run_name: str = Data.settings.get('run_name', 'default').replace(' ', '_')
        filename = run_name + "_" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+".pkl"
        file: Path = Path(directory, filename)
        with file.open("wb") as f:
            pickle.dump(Data, f)

        if not no_plot:
            plot_sampler_comp(file, Data, comment, plotting_settings)

        return Data

    except KeyboardInterrupt as e:
        shell_print(f"\nCaught KeyboardInterrupt — stopping workers: {e}")
        end_all_integrators()
        raise
    except Exception as e:
        shell_print(f"\nCaught Exception — stopping workers: {e}")
        print_exc()
        end_all_integrators()
        raise
    finally:
        end_all_integrators()


def plot_sampler_comp(file: str,
                      Data: SamplerCompData,
                      comment: str = "",
                      plotting_settings: Dict[str, Any] = dict()) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    file: Path = verify_path(file, suffix=".pkl")
    if not isinstance(Data, SamplerCompData):
        raise ValueError(f"Expected a SamplerCompData object in the file, but got {type(Data)}")

    shell_print(f"Plotting data from '{file}'")

    sep = "-"
    width = 60
    line = width * sep + "\n"
    directory = file.parent
    filename = file.stem
    run_name = Data.settings.get('run_name', 'default')
    madnis_tprog = Data.training_progress[MADNIS_KEY]

    with Path(directory, filename + "_summary.txt").open("w") as f:
        f.write(f"Comment: {comment} \n")
        f.write(line)
        f.write(f"{' Run Parameters ':{'#'}^{width}}\n")
        f.write(line)
        f.write(f"run_name={run_name}\n")
        f.write(f"n_cores={Data.integrand_kwargs['n_cores']}\n")
        f.write(f"batch_size={Data.madnis_kwargs['batch_size']}\n")
        f.write(f"training_steps={max(madnis_tprog.steps_losses)}\n")
        f.write(
            f"Discrete Model: {Data.madnis_kwargs['discrete_model']}\n"
        )
        f.write(f"Learned phase: {Data.integrand_kwargs['training_phase'].upper()}\n")
        f.write(f"Target:")
        if Data.target.real_mean:
            f.write(f"\n    RE : {Data.target.real_mean:.8e}")
            if Data.target.real_error:
                f.write(f" +- {Data.target.real_error:.8e}, RSD = {Data.target.real_rsd:.3f}")
        if Data.target.imag_mean:
            f.write(f"\n    IM : {Data.target.imag_mean:.8e}")
            if Data.target.imag_error:
                f.write(f" +- {Data.target.imag_error:.8e}, RSD = {Data.target.imag_rsd:.3f}")

        for identifier, obs in Data.result.items():
            f.write(f"\n\n{line}")
            f.write(f"{f' {identifier} Results ':{'#'}^{width}}\n")
            if obs.real_error:
                f.write(
                    f"    RE : {obs.real_mean:.8e} +- {obs.real_error:.8e}, RSD = {obs.real_rsd:.3f}, ARSD = {obs.abs_real_rsd:.3f}\n")
            if obs.imag_error:
                f.write(
                    f"    IM : {obs.imag_mean:.8e} +- {obs.imag_error:.8e}, RSD = {obs.imag_rsd:.3f}, ARSD = {obs.abs_imag_rsd:.3f}\n")
            f.write(f"Relative Time-Variance: ")
            if obs.real_tvar:
                f.write(f"RE = {obs.real_tvar:.3e}  ")
            if obs.imag_tvar:
                f.write(f"IM = {obs.imag_tvar:.3e}  ")
            f.write(f"\nAbsolute Time-Variance: ")
            if obs.abs_real_tvar:
                f.write(f"RE = {obs.abs_real_tvar:.3e}  ")
            if obs.abs_imag_tvar:
                f.write(f"IM = {obs.abs_imag_tvar:.3e}  ")
            f.write(f"\nCPU-microsecond per sample: {obs.total_time / obs.n_points * 1e6:.3e}\n")
            f.write(f"Number of samples: {obs.n_points}")

    # Plotting setup
    plot_discrete = len(madnis_tprog.discrete_probs) > 0
    n_show = plotting_settings.get("n_show_best_worst_discrete", 3)
    COLS: Dict[str, str] = {
        MADNIS_KEY: "black",
        NAIVE_KEY: "green",
        VEGAS_KEY: "orange",
        HAVANA_KEY: "blue",
    }

    n_spl = len(Data.training_progress)
    fig_snapshot, axs_snapshot = plt.subplots(
        4, 1, sharex=True, layout="constrained", height_ratios=(3, 1, 1, 1), figsize=(10, 8))
    axs_snapshot: list[plt.Axes]
    axs_snapshot[0].set_ylabel("loss")
    axs_snapshot[1].set_ylabel("RSD")
    axs_snapshot[2].set_ylabel("ARSD")
    axs_snapshot[3].set_ylabel("TVAR")
    axs_snapshot[3].set_xlabel("Training Samples")
    for ax in axs_snapshot:
        ax.set_yscale("log")

    # Create secondary x-axis on top of axs_snapshot[0] with training steps
    ax_top = axs_snapshot[0].twiny()
    # Select a sensible number of ticks evenly spaced
    max_ticks = 6
    step_indices = np.linspace(0, len(madnis_tprog.steps_losses) - 1, max_ticks, dtype=int)
    ax_top.set_xticks(np.array(madnis_tprog.nspl_losses)[step_indices])
    ax_top.set_xticklabels([str(madnis_tprog.steps_losses[i]) for i in step_indices])
    ax_top.set_xlabel("Training Steps")

    if plot_discrete:
        fig_discrete, axs_discrete = plt.subplots(
            n_spl, 1, sharex=True, layout="constrained", figsize=(10, n_spl*3 + 0.5))
        if n_spl == 1:
            axs_discrete = [axs_discrete]
        axs_discrete: list[plt.Axes]
        # Create secondary x-axis on top of axs_snapshot[0] with training steps
        axs_discrete[-1].set_xlabel("Training Samples")
        ax_top_discrete = axs_discrete[0].twiny()
        # Select a sensible number of ticks evenly spaced
        max_ticks = 6
        step_indices = np.linspace(0, len(madnis_tprog.steps_discrete) - 1, max_ticks, dtype=int)
        ax_top_discrete.set_xticks(np.array(madnis_tprog.nspl_discrete)[step_indices])
        ax_top_discrete.set_xticklabels([str(madnis_tprog.steps_discrete[i]) for i in step_indices])
        ax_top_discrete.set_xlabel("Training Steps")

    for i, (KEY, tp) in enumerate(Data.training_progress.items()):
        c = COLS.get(KEY) or "black"
        rsds, arsds, tvars = np.array(tp.rsds), np.array(tp.abs_rsds), np.array(tp.tvars)
        nspl_snapshot = np.array(tp.nspl_snapshot)
        discrete_probs = np.array(tp.discrete_probs)
        nspl_discrete = np.array(tp.nspl_discrete)
        if KEY == MADNIS_KEY:
            axs_snapshot[0].plot(np.array(tp.nspl_losses), np.array(tp.losses), color=c, label=KEY)
        axs_snapshot[1].scatter(nspl_snapshot, rsds, color=c, label=KEY)
        axs_snapshot[2].scatter(nspl_snapshot, arsds, color=c, label=KEY)
        axs_snapshot[3].scatter(nspl_snapshot, tvars, color=c, label=KEY)

        if len(discrete_probs) > 0:
            sorted_indices = np.argsort(discrete_probs[-1])[::-1]
            if sorted_indices.size > 2*n_show:
                show_indices = sorted_indices[
                    np.r_[0:n_show, -n_show:0]
                ]
            else:
                show_indices = sorted_indices
            show_probs = discrete_probs.T[show_indices]
            show_channels = madnis_tprog.all_channels[show_indices]

            channel_labels = [" ".join(str(digit) for digit in ch) for ch in show_channels]
            cmap = colors.LinearSegmentedColormap.from_list(
                "core_scaling",
                ["#2b83ba", "#5ab4ac", "#abdda4", "#fdae61", "#d7191c"],
            )
            cols = [
                colors.to_hex(cmap(t))
                for t in np.linspace(0.0, 1.0, len(channel_labels), endpoint=True)
            ]
            ax = axs_discrete[i]
            for label, probs, col in zip(channel_labels, show_probs, cols):
                ax.plot(nspl_discrete, probs, label=label, color=col)
            ax.set_ylabel(KEY)
            ax.legend()

    # Keep top-axis ticks aligned with the primary x-axis data coordinates.
    ax_top.set_xlim(axs_snapshot[0].get_xlim())

    fig_snapshot.suptitle(f"Training progression for {run_name}", y=1.03)

    handles, labels = axs_snapshot[-1].get_legend_handles_labels()
    if handles and n_spl > 1:
        axs_snapshot[0].legend(
            handles,
            labels,
            loc="upper right",
        )

    fig_snapshot.savefig(
        Path(directory, filename + "_training_prog.png"), dpi=300, bbox_inches="tight"
    )
    if plot_discrete:
        ax_top_discrete.set_xlim(axs_discrete[0].get_xlim())
        fig_discrete.suptitle(f"Discrete Probabilities progression for {run_name}")
        fig_discrete.savefig(
            Path(directory, filename + "_discrete_probs.png"), dpi=300, bbox_inches="tight"
        )

    n_spl = len(Data.observables)
    if n_spl > 1:
        fig, axs = plt.subplots(4, 2, sharex=True, layout="constrained",
                                height_ratios=(3, 1, 1, 1), figsize=(10, 8))
        axs: NDArray[plt.Axes]
        # Column labels
        axs[0, 0].set_title("RE")
        axs[0, 1].set_title("IM")
        # Row labels
        axs[0, 0].set_ylabel("I(f)")
        axs[1, 0].set_ylabel("RSD")
        axs[2, 0].set_ylabel("ARSD")
        axs[3, 0].set_ylabel("TVAR")
        for i in range(2):
            axs[3, i].set_xticks(range(n_spl), list(Data.observables.keys()), rotation=45)

        tgt_line_len = n_spl - 1
        tgt = Data.target
        if tgt.real_mean:
            axs[0, 0].hlines(tgt.real_mean, 0, tgt_line_len, color='red')
        if tgt.real_error:
            axs[0, 0].fill_between(
                [0, tgt_line_len],
                [tgt.real_mean - tgt.real_error],
                [tgt.real_mean + tgt.real_error],
                color='red', alpha=0.3
            )
        if tgt.real_rsd:
            axs[1, 0].hlines(tgt.real_rsd, 0, tgt_line_len, color='red')
        if tgt.abs_real_rsd:
            axs[2, 0].hlines(tgt.abs_real_rsd, 0, tgt_line_len, color='red')
        if tgt.real_tvar:
            axs[3, 0].hlines(tgt.real_tvar, 0, tgt_line_len, color='red')

        if tgt.imag_mean:
            axs[0, 1].hlines(tgt.imag_mean, 0, tgt_line_len, color='red')
        if tgt.imag_error:
            axs[0, 1].fill_between(
                [0, tgt_line_len],
                [tgt.imag_mean - tgt.imag_error],
                [tgt.imag_mean + tgt.imag_error],
                color='red', alpha=0.3
            )
        if tgt.imag_rsd:
            axs[1, 1].hlines(tgt.imag_rsd, 0, tgt_line_len, color='red')
        if tgt.abs_imag_rsd:
            axs[2, 1].hlines(tgt.abs_imag_rsd, 0, tgt_line_len, color='red')
        if tgt.imag_tvar:
            axs[3, 1].hlines(tgt.imag_tvar, 0, tgt_line_len, color='red')

        for i, obs in enumerate(Data.result.values()):
            if obs.real_error > 0:
                axs[0, 0].errorbar(i, obs.real_mean, yerr=obs.real_error,
                                   marker='o', markersize=5, capsize=5, color='black')
                axs[1, 0].scatter(i, obs.real_rsd, color='black')
                axs[2, 0].scatter(i, obs.abs_real_rsd, color='black')
                axs[3, 0].scatter(i, obs.real_tvar, color='black')
                for j in range(1, 4):
                    axs[j, 0].set_yscale("log")
            if obs.imag_error > 0:
                axs[0, 1].errorbar(i, obs.imag_mean, yerr=obs.imag_error,
                                   marker='o', markersize=5, capsize=5, color='black')
                axs[1, 1].scatter(i, obs.imag_rsd, color='black')
                axs[2, 1].scatter(i, obs.abs_imag_rsd, color='black')
                axs[3, 1].scatter(i, obs.imag_tvar, color='black')
                for j in range(1, 4):
                    axs[j, 1].set_yscale("log")
        fig.suptitle(f"Integration results for {run_name}")
        fig.savefig(
            Path(directory, filename + "_integration_result.png"), dpi=300, bbox_inches="tight"
        )
