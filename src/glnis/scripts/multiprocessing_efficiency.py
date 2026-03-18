# type: ignore
import pickle
from numpy.typing import NDArray
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path

from glnis.utils.helpers import shell_print, verify_path
from glnis.core.accumulator import GraphProperties
from glnis.scripts.sampler_comparison import SamplerCompData


class MPEfficiencyData:
    def __init__(self,
                 data: List[Tuple[int, int, Dict[str, float]]],) -> None:
        self.samples: List[MPEfficiencyData.Times] = [self.Times[d] for d in data]

    @dataclass
    class Times:
        n_cores: int
        n_samples: int
        times: Dict[str, float]


def run_multiprocessing_efficiency(
    settings_file: str,
    state_file: str = "",
    comment: str = "",
    no_output: bool = False,
    no_plot: bool = False,
    only_plot: bool = False,
    subfolder: str = "multiprocessing_efficiency",
) -> None:

    if only_plot or Path(settings_file).suffix == ".pkl":
        plot_multiprocessing_efficiency(settings_file, comment)
        quit()

    import os
    import signal
    import numpy as np
    import torch

    from glnis.core.integrator import (
        Integrator,
        MadnisIntegrator,
    )
    from glnis.core.parser import SettingsParser
    from madnis.integrator import Integrator as MadNIS

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        shell_print(f"Working on settings {settings_file}")
        Settings = SettingsParser(settings_file)

        # Slice parameters
        scripts: Dict[str, Any] = Settings.settings.get("scripts", dict())
        params: Dict[str, Any] = scripts.get("multiprocessing_efficiency", dict())
        n_cores: List[int] = params.get("n_cores", [1, 2, 4, 8, 16, 32, 64, 128, 256])
        n_samples: List[int] = params.get("n_samples", [1_000, 10_000, 100_000, 1_000_000, 10_000_000])
        max_samples_per_core: int = params.get("max_samples_per_core", 100_000)

        state_file: Path = verify_path(state_file)
        with state_file.open('rb') as f:
            SData: SamplerCompData = torch.load(f, weights_only=False)

        madnis_state = SData.integrator_states.get("MadNIS", None)

        if not no_output:
            OUTPUT_DIR = verify_path("outputs")
            directory = Path(OUTPUT_DIR, Settings.settings['run_name'].replace(" ", "_"), subfolder)
            if not os.path.exists(str(directory)):
                os.makedirs(str(directory))
                shell_print(f"Created output folder at {directory}")
            shell_print(f"Output will be at {directory}")

        Settings.settings["layered_integrator"]["integrator_type"] = "madnis"
        madnis_kwargs = Settings.get_integrator_kwargs()
        integrand_kwargs = Settings.get_integrand_kwargs()
        param_kwargs = Settings.get_parameterisation_kwargs()
        madnis_integrator: MadnisIntegrator = Integrator.from_settings_file(
            settings_file
        )
        integrand = madnis_integrator.integrand
        if madnis_state is None:
            print(
                f"WARNING: Could not find MadNIS state in state file at '{state_file}'. Will use untrained MadNIS instance.")
        else:
            madnis_integrator.import_state(madnis_state)
            print("Successfully imported madnis state (shirley)")

        # Will hold integration results to write to text file and plot
        Data = SlicePlotData(graph_properties=integrand.graph_properties,
                             settings=Settings.settings,
                             madnis_kwargs=madnis_kwargs,
                             integrand_kwargs=integrand_kwargs,
                             param_kwargs=param_kwargs,)

        madnis: MadNIS = madnis_integrator.madnis
        rng = np.random.default_rng(seed)
        EPS = 1e-6

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


def plot_multiprocessing_efficiency(file: str, comment: str = "") -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    from numpy.typing import NDArray

    file: Path = verify_path(file)
    if not file.is_absolute():
        PROJECT_ROOT = Path(__file__).parents[3]
        file = Path(PROJECT_ROOT, file)
    if not file.exists():
        raise FileNotFoundError(
            f"Unable to find pickled object at '{file}'. Path must be either absolute, or relative to the glnis root folder.")
    with file.open('rb') as f:
        Data: MPEfficiencyData = pickle.load(f)

    shell_print(f"Plotting data from '{file}'")

    directory = file.parent
    filename = file.stem
