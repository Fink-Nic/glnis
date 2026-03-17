# type: ignore
import pickle
from typing import Dict, List, Any
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path

from glnis.utils.helpers import shell_print, verify_path
from glnis.core.accumulator import GraphProperties


class SamplerCompData:
    def __init__(self,
                 integrator_identifiers: List[str],
                 graph_properties: GraphProperties,
                 settings: Dict[str, Any] = dict(),
                 madnis_kwargs: Dict[str, Any] = dict(),
                 integrand_kwargs: Dict[str, Any] = dict(),
                 param_kwargs: Dict[str, Any] = dict(),) -> None:
        self.observables: Dict[str, SamplerCompData.Observables] = dict()
        for name in integrator_identifiers:
            self.observables[name] = self.Observables()
        self.graph_properties = graph_properties
        self.settings: Dict[str, Any] = settings
        self.madnis_kwargs: Dict[str, Any] = madnis_kwargs
        self.integrand_kwargs: Dict[str, Any] = integrand_kwargs
        self.param_kwargs: Dict[str, Any] = param_kwargs
        self.plottables: SamplerCompData.Plottables = self.Plottables()
        self.target: SamplerCompData.Observables = self.Observables()
        self.integrator_states: Dict[str, Any] = dict()
        for name in integrator_identifiers:
            self.integrator_states[name] = None

    @dataclass
    class Observables:
        real_central_value: float = 0
        imag_central_value: float = 0
        real_error: float = 0
        imag_error: float = 0
        real_rsd: float = 0
        imag_rsd: float = 0
        n_points: int = 0
        real_tvar: float = 0
        imag_tvar: float = 0
        processing_times: Dict[str, float] = field(default_factory=dict)

        def calc_rsds(self) -> None:
            if not self.real_central_value == 0:
                self.real_rsd = abs(self.real_error / self.real_central_value * self.n_points**0.5)
            if not self.imag_central_value == 0:
                self.imag_rsd = abs(self.imag_error / self.imag_central_value * self.n_points**0.5)

        def calc_tvar(self) -> None:
            if self.n_points > 0:
                total_time = self.processing_times.get('total_time', 0)
                self.real_tvar = self.real_rsd**2 * total_time / self.n_points
                self.imag_tvar = self.imag_rsd**2 * total_time / self.n_points

    @dataclass
    class Plottables:
        losses: List[float] = field(default_factory=list)
        rsds: List[float] = field(default_factory=list)
        tvars: List[float] = field(default_factory=list)
        steps_losses: List[int] = field(default_factory=list)
        steps_snapshot: List[int] = field(default_factory=list)
        means: List[float] = field(default_factory=list)
        errors: List[float] = field(default_factory=list)


def run_sampler_comp(
    file: str,
    comment: str = "",
    no_naive: bool = False,
    no_vegas: bool = False,
    no_havana: bool = False,
    no_output: bool = False,
    no_plot: bool = False,
    only_plot: bool = False,
    export_states: bool = True,
    subfolder: str = "sampler_comp",
) -> None:

    if only_plot or Path(file).suffix == ".pkl":
        plot_sampler_comp(file, comment)
        quit()

    import math
    import os
    import signal
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
    try:
        shell_print(f"Working on settings {file}")
        Settings = SettingsParser(file)

        # Training parameters
        params = Settings.settings["scripts"]["sampler_comp"]
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
            directory = Path(PROJECT_ROOT, OUTPUT_DIR, Settings.settings['run_name'].replace(" ", "_"), subfolder)
            if not os.path.exists(str(directory)):
                os.makedirs(str(directory))
                shell_print(f"Created output folder at {directory}")
            shell_print(f"Output will be at {directory}")

        # Callback for the madnis integrator
        def callback(status) -> None:
            step = status.step + 1
            if step % n_log == 0:
                shell_print(
                    f"Step {step}: loss={status.loss:.5f}, lr = {status.learning_rate:.2e}"
                )
            if step % n_plot_loss == 0:
                Data.plottables.losses.append(status.loss)
                Data.plottables.steps_losses.append(step)
            if step % n_plot_rsd == 0:
                output = madnis_integrator.integrate(n_samples, progress_report=False)
                obs = output.get_observables()
                phase = integrand.training_phase.lower()
                res = obs[f"{phase}_central_value"]
                err = obs[f"{phase}_error"]
                rsd = obs[f"{phase}_rsd"]
                tvar = rsd**2 * obs["processing_times"]["total_time"] / obs["n_points"]
                shell_print(
                    f"""Trained Result after {step} steps of {
                        madnis_integrator.batch_size}, using {n_samples} samples:""",
                    f"    {res:.8e} +- {err:.8e}, RSD = {rsd:.3f}"
                )
                Data.plottables.means.append(res)
                Data.plottables.errors.append(err)
                Data.plottables.rsds.append(rsd)
                Data.plottables.tvars.append(tvar)
                Data.plottables.steps_snapshot.append(step)

        time_last = time()
        integrators: dict[str, Integrator] = dict()
        Settings.settings["layered_integrator"]["integrator_type"] = "madnis"
        madnis_kwargs = Settings.get_integrator_kwargs()
        integrand_kwargs = Settings.get_integrand_kwargs()
        param_kwargs = Settings.get_parameterisation_kwargs()
        madnis_integrator: MadnisIntegrator = Integrator.from_settings_file(
            file
        )
        madnis_integrator.callback = callback
        integrand = madnis_integrator.integrand
        n_total_training_samples = n_training_steps * madnis_kwargs["batch_size"]
        neval = int(n_total_training_samples / nitn)

        if not no_naive:
            Settings.settings["layered_integrator"]["integrator_type"] = "naive"
            integrators["Naive"] = NaiveIntegrator(integrand, **Settings.get_integrator_kwargs())
        if not no_vegas:
            Settings.settings["layered_integrator"]["integrator_type"] = "vegas"
            integrators["Vegas"] = VegasIntegrator(integrand, **Settings.get_integrator_kwargs())
        if not no_havana:
            Settings.settings["layered_integrator"]["integrator_type"] = "havana"
            integrators["Havana"] = HavanaIntegrator(integrand, **Settings.get_integrator_kwargs())

        integrators["MadNIS"] = madnis_integrator

        # Will hold integration results to write to text file and plot
        Data = SamplerCompData(integrator_identifiers=list(integrators.keys()),
                               graph_properties=integrand.graph_properties,
                               settings=Settings.settings,
                               madnis_kwargs=madnis_kwargs,
                               integrand_kwargs=integrand_kwargs,
                               param_kwargs=param_kwargs,)

        shell_print(
            f"""Initializing the Integrand and Integrators took {
                -time_last + (time_last := time()):.2f}s"""
        )
        shell_print(f"MadNIS is using device: {madnis_integrator.madnis.dummy.device}")
        shell_print(f"MadNIS is using scheduler: {madnis_integrator.madnis.scheduler}")
        shell_print(f"Integrand discrete dims: {integrand.discrete_dims}")

        # Parse GammaLoop results
        gl_res = Settings.get_gammaloop_integration_result()
        if gl_res is None:
            Data.target.real_central_value = integrand.target_real
            Data.target.imag_central_value = integrand.target_imag
        else:
            RE_OR_IM = (
                "re" if madnis_integrator.integrand.training_phase == 'real' else 'im'
            )
            gl_int = gl_res['result'][RE_OR_IM]
            gl_err = gl_res['error'][RE_OR_IM]
            gl_neval = gl_res['neval']
            gl_rsd = abs(gl_err / gl_int) * math.sqrt(gl_neval)
            Data.target.real_central_value = gl_res['result']['re']
            Data.target.real_error = gl_res['error']['re']
            Data.target.imag_central_value = gl_res['result']['im']
            Data.target.imag_error = gl_res['error']['im']
            Data.target.n_points = gl_neval
            Data.target.calc_rsds()

            shell_print(
                f"Gammaloop Result: {gl_int:.8g} +- {gl_err:.8g}, RSD = {gl_rsd:.3f}"
            )

        # Training all the integrators
        if not no_vegas:
            shell_print("Training Vegas:")
            vegas_training_result = integrators["Vegas"].train(nitn, neval)
            shell_print(vegas_training_result.summary())
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
            acc = integrator.integrate(n_samples_after_training)
            obs = acc.get_observables()
            for f in fields(Data.observables[identifier]):
                setattr(Data.observables[identifier], f.name, obs.get(f.name, 0))
            Data.observables[identifier].calc_tvar()

        # IMPORTANT: close the worker functions, or your script will hang
        integrand.end()

        if no_output:
            quit()

        run_name = Data.settings['run_name'].replace(' ', '_')
        filename = run_name + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+".pkl"
        file = Path(directory, filename)
        with file.open("wb") as f:
            save(Data, f)

        if no_plot:
            quit()

        plot_sampler_comp(file, comment)

    except KeyboardInterrupt:
        shell_print("\nCaught KeyboardInterrupt — stopping workers.")
        integrand.end()
    finally:
        integrand.end()


def plot_sampler_comp(file: str, comment: str = "") -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.typing import NDArray
    from torch import load

    file: Path = verify_path(file, suffix=".pkl")
    with file.open('rb') as f:
        Data: SamplerCompData = load(f, weights_only=False)

    shell_print(f"Plotting data from '{file}'")

    sep = "-"
    width = 60
    line = width * sep + "\n"
    directory = file.parent
    filename = file.stem

    print(Data.graph_properties)

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
            f.write(f"Relative Time-Variance: RE = {obs.real_tvar:.3e}, IM = {obs.imag_tvar:.3e}\n")
            f.write(f"CPU-microsecond per sample: {obs.processing_times['total_time'] / obs.n_points * 1e6:.3e}\n")
            f.write(f"Number of samples: {obs.n_points}")

    # Plotting setup
    losses, steps_losses = np.array(Data.plottables.losses), np.array(Data.plottables.steps_losses)
    rsds, steps_snapshot = np.array(Data.plottables.rsds), np.array(Data.plottables.steps_snapshot)
    tvars = np.array(Data.plottables.tvars)

    fig, axs = plt.subplots(3, 1, sharex=True, layout="constrained")
    axs: list[plt.Axes]
    axs[0].plot(steps_losses, losses, color="black")
    axs[0].set_ylabel("loss")
    axs[1].scatter(steps_snapshot, rsds, color="black")
    axs[1].set_ylabel("RSD")
    axs[2].scatter(steps_snapshot, tvars, color="black")
    axs[2].set_ylabel("RTVAR")
    axs[2].set_xlabel("Training steps")
    for ax in axs:
        ax.set_yscale("log")
    fig.suptitle(f"MadNIS training progression for {Data.settings['run_name']}")
    plt.savefig(
        Path(directory, filename + "_training_prog.png"), dpi=300, bbox_inches="tight"
    )

    if len(Data.observables) < 2:
        quit()

    fig, axs = plt.subplots(3, 2, sharex=True, layout="constrained",
                            height_ratios=(2, 1, 1), figsize=(10, 8))
    axs: NDArray[plt.Axes]
    # Column labels
    axs[0, 0].set_title("RE")
    axs[0, 1].set_title("IM")
    # Row labels
    axs[0, 0].set_ylabel("I(f)")
    axs[1, 0].set_ylabel("RSD")
    axs[2, 0].set_ylabel("RTVAR")
    for i in range(2):
        axs[2, i].set_xticks(range(len(Data.observables)), list(Data.observables.keys()), rotation=45)
        axs[1, i].set_yscale("log")
        axs[2, i].set_yscale("log")

    axs[0, 0].hlines(Data.target.real_central_value, 0, len(Data.observables)-1, color='red')
    axs[0, 1].hlines(Data.target.imag_central_value, 0, len(Data.observables)-1, color='red')
    for i, obs in enumerate(Data.observables.values()):
        if obs.real_error > 0:
            axs[0, 0].errorbar(i, obs.real_central_value, yerr=obs.real_error,
                               marker='o', markersize=5, capsize=5, color='black')
            axs[1, 0].scatter(i, obs.real_rsd, color='black')
            axs[2, 0].scatter(i, obs.real_tvar, color='black')
        if obs.imag_error > 0:
            axs[0, 1].errorbar(i, obs.imag_central_value, yerr=obs.imag_error,
                               marker='o', markersize=5, capsize=5, color='black')
            axs[1, 1].scatter(i, obs.imag_rsd, color='black')
            axs[2, 1].scatter(i, obs.imag_tvar, color='black')
    fig.suptitle(f"Integration results for {Data.settings['run_name']}")
    fig.savefig(
        Path(directory, filename + "_integration_result.png"), dpi=300, bbox_inches="tight"
    )
