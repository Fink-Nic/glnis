# type: ignore
import pickle
from numpy.typing import NDArray
from typing import Dict, List, Any
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path

from glnis.utils.helpers import shell_print, verify_path
from glnis.core.accumulator import GraphProperties
from glnis.scripts.sampler_comparison import SamplerCompData


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
        t: NDArray
        func_val: NDArray
        prob: NDArray


def run_slice_plots(
    settings_file: str,
    state_file: str,
    comment: str = "",
    no_output: bool = False,
    no_plot: bool = False,
    only_plot: bool = False,
    subfolder: str = "slice_plots",
) -> None:

    if only_plot or Path(settings_file).suffix == ".pkl":
        plot_slices(file, comment)
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

        # Training parameters
        params: Dict[str, Any] = Settings.settings["scripts"]["slice_plots"]
        n_samples_1d = params.get("n_samples_1d", 1000)
        n_samples_2d = params.get("n_samples_2d", 1000)
        n_slices_1d = params.get("n_slices_1d", 2)
        slice_dims_2d = params.get("slice_dims_2d", [[0, 1]])
        seed = params.get("seed", 42)

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
        for _ in range(n_slices_1d):
            # Sample a random discrete point
            if integrand.discrete_dims:
                discrete = [rng.integers(0, ddim) for ddim in integrand.discrete_dims]
                discrete = np.tile(np.array(discrete, dtype=np.uint64), (n_samples_1d, 1))
            else:
                discrete = np.empty((n_samples_1d, 0), dtype=np.uint64)
            # Sample a random direction in the continuous space
            origin = rng.uniform(0.1, 0.9, size=integrand.continuous_dim)
            direction = rng.uniform(0, 1, size=integrand.continuous_dim)
            direction /= np.linalg.norm(direction)
            t_max = np.min((1 - origin) / direction)
            t_min = -np.min(origin / direction)
            t = np.linspace(t_min, t_max, n_samples_1d)
            t *= (1 - EPS)  # Avoid numerical issues at the boundaries
            continuous = origin[None, :] + direction[None, :] * t.reshape(-1, 1)
            # Get the slice data
            layer_input = madnis_integrator.init_layer_data(n_samples_1d)
            layer_input.continuous = continuous
            layer_input.discrete = discrete
            acc = integrand.eval_integrand(layer_input, "training")
            func_val = acc.modules[-1].training_result[0].ravel()
            # Molest MadNIS to get the probabilities for the slice
            x_all = torch.from_numpy(np.hstack([discrete, continuous])).to(madnis_integrator.device)
            prob = madnis.flow.prob(x_all).numpy(force=True).ravel()
            Data.slices1d.append(SlicePlotData.Slice(t=t, func_val=func_val, prob=prob))

        for dims in slice_dims_2d:
            # Sample a random discrete point
            if integrand.discrete_dims:
                discrete = [rng.integers(0, ddim) for ddim in integrand.discrete_dims]
                discrete = np.tile(np.array(discrete, dtype=np.uint64), (n_samples_2d**2, 1))
            else:
                discrete = np.empty((n_samples_2d**2, 0), dtype=np.uint64)
            # Sample a random point and create a grid in the 2 selected dimensions
            point = rng.uniform(EPS, 1 - EPS, size=integrand.continuous_dim)
            grid_1d = np.linspace(EPS, 1 - EPS, n_samples_2d)
            grid_2d = np.stack(arrays=np.meshgrid(grid_1d, grid_1d), axis=-1).reshape(-1, 2)
            continuous = np.tile(point, (n_samples_2d**2, 1))
            continuous[:, dims] = grid_2d

            # Get the slice data
            layer_input = madnis_integrator.init_layer_data(n_samples_2d**2)
            layer_input.continuous = continuous
            layer_input.discrete = discrete
            acc = integrand.eval_integrand(layer_input, "training")
            func_val = acc.modules[-1].training_result[0].reshape(n_samples_2d, n_samples_2d)
            # Molest MadNIS to get the probabilities for the slice
            x_all = torch.from_numpy(np.hstack([discrete, continuous])).to(madnis_integrator.device)
            prob = madnis.flow.prob(x_all).numpy(force=True).reshape(n_samples_2d, n_samples_2d)
            Data.slices2d.append(SlicePlotData.Slice(t=grid_2d, func_val=func_val, prob=prob))

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
        Data: SlicePlotData = pickle.load(f)

    shell_print(f"Plotting data from '{file}'")

    directory = file.parent
    filename = file.stem
    EPS = 1e-6

    for i, slice in enumerate(Data.slices1d):
        slice: SlicePlotData.Slice
        fig, axs = plt.subplots(3, 1, sharex=True, layout="constrained",
                                height_ratios=[1, 0.3, 0.3], figsize=(6, 8))
        axs: List[plt.Axes]
        axs[0].plot(slice.t, np.abs(slice.func_val), label="|Integrand|")
        axs[0].plot(slice.t, np.abs(slice.prob), label="|Probability|")
        axs[0].legend()
        axs[1].plot(slice.t, np.abs(slice.func_val) / np.abs(slice.prob))
        axs[2].plot(slice.t, np.sign(slice.func_val) * np.sign(slice.prob))
        # axs[0].set_ylabel("|Integrand Value|")
        axs[1].set_ylabel("|Ratio|")
        axs[2].set_ylabel("Sign match")
        axs[0].set_yscale("log")
        axs[1].set_yscale("log")
        axs[2].set_yticks([-1, 0, 1])
        axs[2].set_yticklabels([r'$\neq$', '0', '='])
        fig.suptitle(f"1D Slices {Data.settings['run_name']}")
        plt.savefig(
            Path(directory, filename + f"_slice1d_{i}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    for i, slice in enumerate(Data.slices2d):
        slice: SlicePlotData.Slice
        data_log_titles = ["|Integrand|", "|Probability|"]
        data_log = [np.abs(slice.func_val), np.abs(slice.prob)]
        data_discrete = np.sign(slice.func_val) * np.sign(slice.prob)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8),
                                 sharex=True, sharey=True, constrained_layout=True)
        (ax1, ax2), (ax3, ax4) = axes

        # Plot the funcval and probability with shared log color scale
        vmin = min(d[d > 0].min() for d in data_log)
        vmax = max(d.max() for d in data_log)
        log_norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        imgs = []
        for ax, data, title in zip([ax1, ax2], data_log, data_log_titles):
            ax: plt.Axes
            im = ax.imshow(data, norm=log_norm, cmap='plasma',
                           origin='lower', extent=[0, 1, 0, 1])
            imgs.append(im)
            ax.set_title(title)

        ratio = np.abs(slice.func_val) / np.abs(slice.prob)
        log_norm_ratio = colors.LogNorm(vmin=ratio[ratio > 0].min(), vmax=ratio.max())
        imratio = ax3.imshow(np.abs(slice.func_val) / np.abs(slice.prob), norm=log_norm_ratio, cmap='plasma',
                             origin='lower', extent=[0, 1, 0, 1])
        ax3.set_title("|Ratio|")

        # Define Normalization for the 4th (Discrete: -1, 0, 1)
        # Boundaries are set at the midpoints to center the colors
        discrete_cmap = colors.ListedColormap(['#e74c3c', '#ecf0f1', '#2ecc71'])  # Red, Grey, Green
        bounds = [-1.5, -0.5, 0.5, 1.5]
        discrete_norm = colors.BoundaryNorm(bounds, discrete_cmap.N)
        ax4: plt.Axes
        im4 = ax4.imshow(data_discrete, cmap=discrete_cmap, norm=discrete_norm,
                         origin='lower', extent=[0, 1, 0, 1])
        # ax4.set_title("Sign match")

        # Add Colorbars
        fraction = 0.046  # Default fraction for colorbar size
        padding = 0.04  # Default padding between plot and colorbar
        # Shared colorbar for the first two, anchor it to the first plot, in order to place it between them
        cbar_f_p = fig.colorbar(imgs[0], ax=ax1, fraction=fraction, pad=padding)

        # Dedicated colorbar for the ratio
        cbar_ratio = fig.colorbar(imratio, ax=ax3, fraction=fraction, pad=padding)

        # Dedicated colorbar for the discrete plot
        cbar_disc = fig.colorbar(im4, ax=ax4, fraction=fraction, pad=padding)
        cbar_disc.set_label('Sign')
        cbar_disc.set_ticks([-1, 0, 1])
        cbar_disc.set_ticklabels([r'$\neq$', '0', '='])

        plt.savefig(
            Path(directory, filename + f"_slice2d_{i}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
