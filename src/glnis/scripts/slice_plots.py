# type: ignore
import pickle
from numpy.typing import NDArray
from typing import Dict, List, Any
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path

from glnis.utils.helpers import shell_print
from glnis.core.accumulator import GraphProperties


class SlicePlotData:
    def __init__(self,
                 graph_properties: GraphProperties,
                 settings: Dict[str, Any] = dict(),
                 madnis_kwargs: Dict[str, Any] = dict(),
                 integrand_kwargs: Dict[str, Any] = dict(),
                 param_kwargs: Dict[str, Any] = dict(),
                 slices1d=[],
                 slices2d=[],) -> None:
        self.graph_properties = graph_properties
        self.settings: Dict[str, Any] = settings
        self.madnis_kwargs: Dict[str, Any] = madnis_kwargs
        self.integrand_kwargs: Dict[str, Any] = integrand_kwargs
        self.param_kwargs: Dict[str, Any] = param_kwargs
        self.slices1d: List[SlicePLotData.Slice] = slices1d
        self.slices2d: List[SlicePLotData.Slice] = slices2d

    @dataclass
    class Slice:
        t: NDArray[float]
        func_val: NDArray[float]
        prob: NDArray[float]


def run_slice_plots(
    file: str,
    comment: str = "",
    no_output: bool = False,
    no_plot: bool = False,
    only_plot: bool = False,
    subfolder: str = "slice_plots",
) -> None:

    if only_plot or Path(file).suffix == ".pkl":
        plot_slices(file, comment)
        quit()

    import math
    import os
    import signal
    from time import time

    from glnis.core.integrator import (
        Integrator,
        MadnisIntegrator,
    )
    from glnis.core.parser import SettingsParser

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        shell_print(f"Working on settings {file}")
        Settings = SettingsParser(file)

        # Training parameters
        params = Settings.settings["scripts"]["slice_plots"]
        n_training_steps = params["n_training_steps"]
        n_samples_1d = params["n_samples_1d"]
        n_samples_2d = params["n_samples_2d"]

        if not no_output:
            PROJECT_ROOT = Path(__file__).parents[3]
            OUTPUT_DIR = "outputs"
            directory = Path(PROJECT_ROOT, OUTPUT_DIR, Settings.settings['run_name'].replace(" ", "_"), subfolder)
            if not os.path.exists(str(directory)):
                os.makedirs(str(directory))
                shell_print(f"Created output folder at {directory}")
            shell_print(f"Output will be at {directory}")

        time_last = time()
        Settings.settings["layered_integrator"]["integrator_type"] = "madnis"
        madnis_kwargs = Settings.get_integrator_kwargs()
        integrand_kwargs = Settings.get_integrand_kwargs()
        param_kwargs = Settings.get_parameterisation_kwargs()
        madnis_integrator: MadnisIntegrator = Integrator.from_settings_file(
            file
        )
        integrand = madnis_integrator.integrand

        # Will hold integration results to write to text file and plot
        Data = SlicePlotData(graph_properties=integrand.graph_properties,
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

        shell_print("Training MadNIS:")
        madnis_integrator.train(n_training_steps, callback)
        shell_print(f"Training MadNIS took {-time_last + (time_last := time()): .2f}s")

        # IMPORTANT: close the worker functions, or your script will hang
        integrand.end()

        if no_output:
            quit()

        run_name = Data.settings['run_name'].replace(' ', '_')
        filename = run_name + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+".pkl"
        file = Path(directory, filename)
        with file.open("wb") as f:
            pickle.dump(Data, f)

        if no_plot:
            quit()

        plot_slices(file, comment)

    except KeyboardInterrupt:
        shell_print("\nCaught KeyboardInterrupt — stopping workers.")
        integrand.end()
    finally:
        integrand.end()


def plot_slices(file: str, comment: str = "") -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.typing import NDArray

    file: Path = Path(file)
    if not file.is_absolute():
        PROJECT_ROOT = Path(__file__).parents[3]
        file = Path(PROJECT_ROOT, file)
    if not file.exists():
        raise FileNotFoundError(
            f"Unable to find pickled object at '{file}'. Path must be either absolute, or relative to the glnis root folder.")
    with file.open('rb') as f:
        Data: SlicePlotData = pickle.load(f)

    shell_print(f"Plotting data from '{file}'")
