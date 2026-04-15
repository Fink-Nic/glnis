# type: ignore
import pickle
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from glnis.utils.helpers import shell_print, verify_path
from glnis.scripts.sampler_comparison import SamplerCompData


class MPEfficiencyData:
    def __init__(self,
                 n_cores: List[int],
                 n_samples: List[int],
                 data: List[Tuple[int, int, Dict[str, float]]] = [],) -> None:
        self.n_cores = n_cores
        self.n_samples = n_samples
        self.data: List[MPEfficiencyData.Times] = [self.Times(d) for d in data]

    @dataclass
    class Times:
        n_cores: int
        n_samples: int
        times: Dict[str, float]


def run_multiprocessing_efficiency(
    file: str,
    comment: str = "",
    no_output: bool = False,
    no_plot: bool = False,
    subroutine: str = "multiprocessing_efficiency",
) -> MPEfficiencyData | None:
    import os
    import signal
    import torch

    from glnis.core.integrator import (
        Integrator,
        MadnisIntegrator,
    )
    from glnis.core.parser import SettingsParser

    signal.signal(signal.SIGINT, signal.default_int_handler)

    file = verify_path(file)
    try:
        with file.open('rb') as f:
            SData = torch.load(f, weights_only=False)
    except:
        with file.open('rb') as f:
            SData = pickle.load(f)

    if isinstance(SData, SamplerCompData):
        pass
    elif isinstance(SData, MPEfficiencyData):
        plot_multiprocessing_efficiency(file, comment)
        quit()
    else:
        raise ValueError(
            f"Expected pickled object at '{file}' to contain either a SamplerCompData or MPEfficiencyData object, but found {type(SData)}.")

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        shell_print(f"Working on settings {file}")
        Settings = SettingsParser(SData.settings)

        # parameters
        scripts: Dict[str, Any] = Settings.settings.get("scripts", dict())
        params: Dict[str, Any] = scripts.get(subroutine, dict())
        n_cores: List[int] = params.get("n_cores", [1, 2, 4, 8, 16, 32, 64, 128, 256])
        n_samples: List[int] = params.get("n_samples", [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000])
        max_samples_per_core: int = params.get("max_samples_per_core", 1_000_000)

        madnis_state = SData.integrator_states.get("MadNIS", None)

        if not no_output:
            OUTPUT_DIR = verify_path("outputs")
            directory = Path(OUTPUT_DIR, Settings.settings['run_name'].replace(" ", "_"), subroutine)
            if not os.path.exists(str(directory)):
                os.makedirs(str(directory))
                shell_print(f"Created output folder at {directory}")
            shell_print(f"Output will be at {directory}")

        Settings.settings["layered_integrand"]["use_prec"] = False
        Settings.settings["layered_integrand"]["use_f128"] = False
        Settings.settings["layered_integrand"]["n_cores"] = max(n_cores)
        Settings.settings["layered_integrator"]["integrator_type"] = "madnis"
        Settings.settings["layered_integrator"]["pretrain_c_flow"] = False
        madnis_integrator: MadnisIntegrator = Integrator.from_settings(
            Settings.settings
        )
        if madnis_state is None:
            print(
                f"WARNING: Could not find MadNIS state at '{file}'. Will use untrained MadNIS instance.")
        else:
            madnis_integrator.import_state(madnis_state)
            print("Successfully imported madnis state")

        # Will hold integration results to write to text file and plot
        Data = MPEfficiencyData(n_cores=n_cores, n_samples=n_samples)

        for cores in n_cores:
            madnis_integrator.integrand.n_cores = cores

            for samples in n_samples:
                if samples / cores > max_samples_per_core:
                    continue
                acc = madnis_integrator.integrate(samples, n_start=10_000_000)
                times = MPEfficiencyData.Times(cores, samples, acc.processing_times.processing_times)
                Data.data.append(times)

        if no_output:
            quit()

        run_name = SData.settings['run_name'].replace(' ', '_')
        filename = run_name + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+".pkl"
        file = Path(directory, filename)
        with file.open("wb") as f:
            pickle.dump(Data, f)

        if no_plot:
            quit()

        plot_multiprocessing_efficiency(file, comment)

    except KeyboardInterrupt:
        shell_print(f"\nCaught KeyboardInterrupt — stopping workers: {e}")
        madnis_integrator.free()
    except Exception as e:
        shell_print(f"\nCaught Exception — stopping workers: {e}")
        from traceback import print_exc
        print_exc()
        madnis_integrator.free()
        raise
    finally:
        madnis_integrator.free()


def plot_multiprocessing_efficiency(file: str, comment: str = "") -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np

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

    width = 1 / 3 / len(Data.n_cores)
    # Build a readable sequential palette from cool blue to warm red.
    cmap = colors.LinearSegmentedColormap.from_list(
        "core_scaling",
        ["#2b83ba", "#5ab4ac", "#abdda4", "#fdae61", "#d7191c"],
    )
    cols = [
        colors.to_hex(cmap(t))
        for t in np.linspace(0.0, 1.0, len(Data.n_cores), endpoint=True)
    ]
    fig, ax = plt.subplots(layout="constrained")
    ax.set_xticks(range(len(Data.n_samples)), Data.n_samples, rotation=45)

    added_labels = []
    for d in Data.data:
        mus_factor = 1.0e6 / d.n_samples
        cidx = Data.n_cores.index(d.n_cores)
        sidx = Data.n_samples.index(d.n_samples)
        offset = 2 * (cidx - (len(Data.n_cores) - 1) / 2) * width
        loc = sidx + offset
        lbl = f"{d.n_cores}"
        if lbl in added_labels:
            ax.bar(loc, mus_factor * d.times['total_time'], width=width, color=cols[cidx])
        else:
            added_labels.append(lbl)
            ax.bar(loc, mus_factor * d.times['total_time'], width=width, color=cols[cidx], label=lbl)

    ax.set_xlabel("Number of Samples")
    ax.set_ylabel(r"$t_{eval}$ [CPU-µs]")
    ax.set_yscale("log")
    ax.legend(title=r"$n_{cores}$", loc="upper right")
    ax.set_title("Total time")
    fig.savefig(
        Path(directory, filename + "_total.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    processing_times = Data.data[0].times
    sampler_key = [k for k in processing_times.keys() if "sampler" in k]
    sampler_key = None if len(sampler_key) == 0 else sampler_key[0]
    param_key = [k for k in processing_times.keys() if "param" in k]
    param_key = None if len(param_key) == 0 else param_key[0]
    integrand_key = [k for k in processing_times.keys() if "integrand" in k]
    integrand_key = None if len(integrand_key) == 0 else integrand_key[0]

    names = ["sampler", "param", "integrand"]
    keys = [sampler_key, param_key, integrand_key]

    for name, key in zip(names, keys):
        if key is None:
            continue
        fig, ax = plt.subplots(layout="constrained")
        ax.set_xticks(range(len(Data.n_samples)), Data.n_samples, rotation=45)

        added_labels = []
        for d in Data.data:
            mus_factor = 1.0e6 / d.n_samples
            cidx = Data.n_cores.index(d.n_cores)
            sidx = Data.n_samples.index(d.n_samples)
            offset = 2 * (cidx - (len(Data.n_cores) - 1) / 2) * width
            loc = sidx + offset
            lbl = f"{d.n_cores}"
            if lbl in added_labels:
                ax.bar(loc, mus_factor * d.times[key], width=width, color=cols[cidx])
            else:
                added_labels.append(lbl)
                ax.bar(loc, mus_factor * d.times[key], width=width, color=cols[cidx], label=lbl)

        ax.set_xlabel("Number of Samples")
        ax.set_ylabel(r"$t_{eval}$ [CPU-µs]")
        ax.set_yscale("log")
        ax.legend(title=r"$n_{cores}$", loc="upper right")
        match name:
            case "sampler":
                ax.set_title(f"Sampler time")
            case "param":
                ax.set_title(f"Parameterisation time")
            case "integrand":
                ax.set_title(f"Integrand time")
        fig.savefig(
            Path(directory, filename + f"_{name}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
