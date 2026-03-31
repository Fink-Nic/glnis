# type: ignore
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from madnis.integrator import TrainingStatus

from glnis.utils.helpers import shell_print, verify_path
from glnis.core.accumulator import GraphProperties, Observables, DefaultAccumulator


class SamplerCompData:
    def __init__(self,
                 integrator_identifiers: List[str],
                 graph_properties: GraphProperties,
                 target: Observables,
                 settings: Dict[str, Any] = dict(),
                 madnis_kwargs: Dict[str, Any] = dict(),
                 madnis_info: Dict[str, Any] = dict(),
                 integrand_kwargs: Dict[str, Any] = dict(),
                 param_kwargs: Dict[str, Any] = dict(),) -> None:
        self.result: Dict[str, Observables] = dict()
        self.observables: Dict[str, Dict[str, Any]]
        for name in integrator_identifiers:
            self.result[name] = Observables()
            self.observables[name] = dict()
        self.graph_properties = graph_properties
        self.target = target
        self.settings: Dict[str, Any] = settings
        self.madnis_info: Dict[str, Any] = madnis_info
        self.madnis_kwargs: Dict[str, Any] = madnis_kwargs
        self.integrand_kwargs: Dict[str, Any] = integrand_kwargs
        self.param_kwargs: Dict[str, Any] = param_kwargs
        self.plottables: SamplerCompData.Plottables = self.Plottables()
        self.integrator_states: Dict[str, Any] = dict()
        for name in integrator_identifiers:
            self.integrator_states[name] = None

    def __setstate__(self, state):
        """Custom unpickling logic to handle changes in the class definition over time."""
        self.__dict__.update(state)

        # Provide defaults for fields that didn't exist when the file was saved
        if 'madnis_info' not in state:
            self.madnis_info = dict()

    @dataclass
    class Plottables:
        losses: List[float] = field(default_factory=list)
        rsds: List[float] = field(default_factory=list)
        tvars: List[float] = field(default_factory=list)
        abs_tvars: List[float] = field(default_factory=list)
        steps_losses: List[int] = field(default_factory=list)
        steps_snapshot: List[int] = field(default_factory=list)
        means: List[float] = field(default_factory=list)
        errors: List[float] = field(default_factory=list)


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
) -> SamplerCompData | None:

    if isinstance(file, dict):
        pass
    elif Path(file).suffix == ".pkl":
        plot_sampler_comp(file, comment)
        quit()

    import os
    import signal
    import gc
    from time import time
    from torch import save

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
        shell_print(f"Working on settings {"dictionary" if isinstance(file, dict) else file}")
        Settings = SettingsParser(file)

        # Training parameters
        scripts: Dict[str, Any] = Settings.settings.get("scripts", dict())
        params: Dict[str, Any] = scripts.get(subroutine, dict())
        n_training_steps = params.get("n_training_steps", 1000)
        n_log = params.get("n_log", 10)
        n_plot_rsd = params.get("n_plot_rsd", 100)
        n_plot_loss = params.get("n_plot_loss", 2)
        n_samples = params.get("n_samples", 10_000)
        n_samples_after_training = params.get("n_samples_after_training", 100_000)
        nitn = params.get('nitn', 10)  # number of vegas/havana training iterations
        # number of evals per vegas/havana training iteration

        if not no_output:
            PROJECT_ROOT = Path(__file__).parents[3]
            OUTPUT_DIR = "outputs"
            directory = Path(PROJECT_ROOT, OUTPUT_DIR, Settings.settings['run_name'].replace(" ", "_"), subroutine)
            if not os.path.exists(str(directory)):
                os.makedirs(str(directory))
                shell_print(f"Created output folder at {directory}")
            shell_print(f"Output will be at {directory}")

        # Callback for the madnis integrator
        def callback(status: TrainingStatus) -> None:
            step = status.step + 1
            if step % n_log == 0:
                shell_print(
                    f"Step {step}: loss={status.loss:.5f}, lr = {status.learning_rate:.2e}"
                )
            if step % n_plot_loss == 0:
                Data.plottables.losses.append(status.loss)
                Data.plottables.steps_losses.append(step)
            if step % n_plot_rsd == 0:
                acc = madnis_integrator.integrate(n_samples, progress_report=False)
                obs = acc.get_observables()
                phase = madnis_integrator.integrand.training_phase.lower()
                Data.plottables.means.append(obs[f"{phase}_central_value"])
                Data.plottables.errors.append(obs[f"{phase}_error"])
                Data.plottables.rsds.append(obs[f"{phase}_rsd"])
                Data.plottables.tvars.append(obs[f"{phase}_tvar"])
                Data.plottables.abs_tvars.append(obs[f"abs_{phase}_tvar"])
                Data.plottables.steps_snapshot.append(step)
                shell_print(f"Trained Result after {step} steps of {madnis_integrator.batch_size}:")
                shell_print(acc.statistics.str_report())

        time_last = time()
        Settings.settings["layered_integrator"]["integrator_type"] = "madnis"
        madnis_kwargs = Settings.get_integrator_kwargs()
        integrand_kwargs = Settings.get_integrand_kwargs()
        param_kwargs = Settings.get_parameterisation_kwargs()
        madnis_integrator: MadnisIntegrator = Integrator.from_settings(
            Settings.settings
        )
        integrators["MadNIS"] = madnis_integrator
        madnis_integrator.callback = callback
        n_total_training_samples = n_training_steps * madnis_kwargs["batch_size"]
        neval = int(n_total_training_samples / nitn)

        if not no_naive:
            Settings.settings["layered_integrator"]["integrator_type"] = "naive"
            integrators["Naive"] = NaiveIntegrator(madnis_integrator.integrand, **Settings.get_integrator_kwargs())
        if not no_vegas:
            Settings.settings["layered_integrator"]["integrator_type"] = "vegas"
            integrators["Vegas"] = VegasIntegrator(madnis_integrator.integrand, **Settings.get_integrator_kwargs())
        if not no_havana:
            Settings.settings["layered_integrator"]["integrator_type"] = "havana"
            integrators["Havana"] = HavanaIntegrator(madnis_integrator.integrand, **Settings.get_integrator_kwargs())

        # Will hold integration results to write to text file and plot
        Data = SamplerCompData(integrator_identifiers=list(integrators.keys()),
                               graph_properties=madnis_integrator.integrand.graph_properties,
                               target=madnis_integrator.integrand.target,
                               settings=Settings.settings,
                               madnis_kwargs=madnis_kwargs,
                               madnis_info=madnis_integrator.get_info(),
                               integrand_kwargs=integrand_kwargs,
                               param_kwargs=param_kwargs,)

        shell_print(
            f"""Initializing the Integrand and Integrators took {
                -time_last + (time_last := time()):.2f}s"""
        )
        madnis_integrator.display_info()

        # Training all the integrators
        if not no_vegas:
            shell_print("Training Vegas:")
            vegas_training_result = integrators["Vegas"].train(nitn, neval)
            shell_print(vegas_training_result)
            shell_print(f"Training Vegas took {-time_last + (time_last := time()): .2f}s")

        if not no_havana:
            shell_print("Training Havana:")
            havana_training_result = integrators["Havana"].train(nitn, neval)
            shell_print(havana_training_result)
            shell_print(f"Training Havana took {-time_last + (time_last := time()): .2f}s")

        shell_print("Training MadNIS:")
        madnis_integrator.train(n_training_steps, callback)
        shell_print(f"Training MadNIS took {-time_last + (time_last := time()): .2f}s")

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

        run_name: str = Data.settings['run_name'].replace(' ', '_')
        filename = run_name + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+".pkl"
        file: Path = Path(directory, filename)
        with file.open("wb") as f:
            save(Data, f)

        if not no_plot:
            plot_sampler_comp(file, comment)

        return Data

    except KeyboardInterrupt as e:
        shell_print(f"\nCaught KeyboardInterrupt — stopping workers: {e}")
        end_all_integrators()
    except Exception as e:
        shell_print(f"\nCaught Exception — stopping workers: {e}")
        end_all_integrators()
    finally:
        end_all_integrators()


def plot_sampler_comp(file: str, comment: str = "") -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.typing import NDArray
    from torch import load

    file: Path = verify_path(file, suffix=".pkl")
    with file.open('rb') as f:
        Data: SamplerCompData = load(f, weights_only=False)
        if not isinstance(Data, SamplerCompData):
            raise ValueError(f"Expected a SamplerCompData object in the file, but got {type(Data)}")

    shell_print(f"Plotting data from '{file}'")

    sep = "-"
    width = 60
    line = width * sep + "\n"
    directory = file.parent
    filename = file.stem

    with Path(directory, filename + "_summary.txt").open("w") as f:
        f.write(f"Comment: {comment} \n")
        f.write(line)
        f.write(f"{' Run Parameters ':{'#'}^{width}}\n")
        f.write(line)
        f.write(f"run_name={Data.settings['run_name']}\n")
        f.write(f"n_cores={Data.integrand_kwargs['n_cores']}\n")
        f.write(f"batch_size={Data.madnis_kwargs['batch_size']}\n")
        f.write(f"training_steps={max(Data.plottables.steps_losses)}\n")
        f.write(
            f"Discrete Model: {Data.madnis_kwargs['discrete_model']}\n"
        )
        f.write(f"Learned phase: {Data.integrand_kwargs['training_phase'].upper()}\n")
        f.write(f"Target:")
        if Data.target.real_central_value:
            f.write(f"\n    RE : {Data.target.real_central_value:.8e}")
            if Data.target.real_error:
                f.write(f" +- {Data.target.real_error:.8e}, RSD = {Data.target.real_rsd:.3f}")
        if Data.target.imag_central_value:
            f.write(f"\n    IM : {Data.target.imag_central_value:.8e}")
            if Data.target.imag_error:
                f.write(f" +- {Data.target.imag_error:.8e}, RSD = {Data.target.imag_rsd:.3f}")

        for identifier, obs in Data.observables.items():
            f.write(f"\n\n{line}")
            f.write(f"{f' {identifier} Results ':{'#'}^{width}}\n")
            if obs.real_error:
                f.write(
                    f"    RE : {obs.real_central_value:.8e} +- {obs.real_error:.8e}, RSD = {obs.real_rsd:.3f}\n")
            if obs.imag_error:
                f.write(
                    f"    IM : {obs.imag_central_value:.8e} +- {obs.imag_error:.8e}, RSD = {obs.imag_rsd:.3f}\n")
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
    losses, steps_losses = np.array(Data.plottables.losses), np.array(Data.plottables.steps_losses)
    rsds, steps_snapshot = np.array(Data.plottables.rsds), np.array(Data.plottables.steps_snapshot)
    tvars, atvars = np.array(Data.plottables.tvars), np.array(Data.plottables.abs_tvars)

    if len(steps_losses) and len(steps_snapshot):
        fig, axs = plt.subplots(4, 1, sharex=True, layout="constrained",
                                height_ratios=(3, 1, 1, 1), figsize=(10, 8))
        axs: list[plt.Axes]
        axs[0].plot(steps_losses, losses, color="black")
        axs[0].set_ylabel("loss")
        axs[1].scatter(steps_snapshot, rsds, color="black")
        axs[1].set_ylabel("RSD")
        axs[2].scatter(steps_snapshot, tvars, color="black")
        axs[2].set_ylabel("TVAR")
        axs[3].scatter(steps_snapshot, atvars, color="black")
        axs[3].set_ylabel("ATVAR")
        axs[3].set_xlabel("Training steps")
        for ax in axs:
            ax.set_yscale("log")
        fig.suptitle(f"MadNIS training progression for {Data.settings['run_name']}")
        plt.savefig(
            Path(directory, filename + "_training_prog.png"), dpi=300, bbox_inches="tight"
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
        axs[2, 0].set_ylabel("TVAR")
        axs[3, 0].set_ylabel("ATVAR")
        for i in range(2):
            axs[3, i].set_xticks(range(n_spl), list(Data.observables.keys()), rotation=45)
            axs[1, i].set_yscale("log")
            axs[2, i].set_yscale("log")
            axs[3, i].set_yscale("log")

        tgt_line_len = n_spl - 1
        tgt = Data.target
        if tgt.real_central_value:
            axs[0, 0].hlines(tgt.real_central_value, 0, tgt_line_len, color='red')
        if tgt.real_rsd:
            axs[1, 0].hlines(tgt.real_rsd, 0, tgt_line_len, color='red')
        if tgt.real_tvar:
            axs[2, 0].hlines(tgt.real_tvar, 0, tgt_line_len, color='red')
        if tgt.abs_real_tvar:
            axs[3, 0].hlines(tgt.abs_real_tvar, 0, tgt_line_len, color='red')

        if tgt.imag_central_value:
            axs[0, 1].hlines(tgt.imag_central_value, 0, tgt_line_len, color='red')
        if tgt.imag_rsd:
            axs[1, 1].hlines(tgt.imag_rsd, 0, tgt_line_len, color='red')
        if tgt.imag_tvar:
            axs[2, 1].hlines(tgt.imag_tvar, 0, tgt_line_len, color='red')
        if tgt.abs_imag_tvar:
            axs[3, 1].hlines(tgt.abs_imag_tvar, 0, tgt_line_len, color='red')

        for i, obs in enumerate(Data.observables.values()):
            if obs.real_error > 0:
                axs[0, 0].errorbar(i, obs.real_central_value, yerr=obs.real_error,
                                   marker='o', markersize=5, capsize=5, color='black')
                axs[1, 0].scatter(i, obs.real_rsd, color='black')
                axs[2, 0].scatter(i, obs.real_tvar, color='black')
                axs[3, 0].scatter(i, obs.abs_real_tvar, color='black')
            if obs.imag_error > 0:
                axs[0, 1].errorbar(i, obs.imag_central_value, yerr=obs.imag_error,
                                   marker='o', markersize=5, capsize=5, color='black')
                axs[1, 1].scatter(i, obs.imag_rsd, color='black')
                axs[2, 1].scatter(i, obs.imag_tvar, color='black')
                axs[3, 1].scatter(i, obs.abs_imag_tvar, color='black')
        fig.suptitle(f"Integration results for {Data.settings['run_name']}")
        fig.savefig(
            Path(directory, filename + "_integration_result.png"), dpi=300, bbox_inches="tight"
        )
