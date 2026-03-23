# type: ignore
from numpy.typing import NDArray
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from glnis.utils.helpers import shell_print, verify_path
from glnis.core.accumulator import GraphProperties, TrainingAccumulator
from glnis.scripts.sampler_comparison import SamplerCompData


class SlicePlotData:
    def __init__(self,
                 graph_properties: GraphProperties,
                 settings: Dict[str, Any] = dict(),
                 madnis_kwargs: Dict[str, Any] = dict(),
                 integrand_kwargs: Dict[str, Any] = dict(),
                 param_kwargs: Dict[str, Any] = dict(),
                 slices1d=[],
                 slices2d=[],
                 EPS: float = 1e-6
                 ) -> None:
        self.graph_properties = graph_properties
        self.settings: Dict[str, Any] = settings
        self.madnis_kwargs: Dict[str, Any] = madnis_kwargs
        self.integrand_kwargs: Dict[str, Any] = integrand_kwargs
        self.param_kwargs: Dict[str, Any] = param_kwargs
        self.slices1d: List[Slice] = slices1d
        self.slices2d: List[Slice] = slices2d
        self.EPS = EPS


@dataclass
class Slice:
    t: NDArray
    func_val: NDArray
    prob: NDArray


def run_slice_plots(
    file: str,
    comment: str = "",
    no_output: bool = False,
    no_plot: bool = False,
    subroutine: str = "slice_plots",
) -> SlicePlotData | None:

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

    file = verify_path(file)
    with file.open('rb') as f:
        SData = torch.load(f, weights_only=False)
    if isinstance(SData, SamplerCompData):
        pass
    elif isinstance(SData, SlicePlotData):
        plot_slices(file, comment)
        quit()
    else:
        raise ValueError(
            f"Unrecognized data type in file '{file}'. Expected either SamplerCompData or SlicePlotData, but got {type(SData)}.")

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        shell_print(f"Working on {file}")
        Settings = SettingsParser(SData.settings)

        # Slice parameters
        scripts: Dict[str, Any] = Settings.settings.get("scripts", dict())
        params: Dict[str, Any] = scripts.get("slice_plots", dict())
        n_samples_1d = params.get("n_samples_1d", 1000)
        n_samples_2d = params.get("n_samples_2d", 100)
        n_slices_1d = params.get("n_slices_1d", 2)
        slice_dims_2d = params.get("slice_dims_2d", [[0, 1]])
        EPS = params.get("EPS", 1e-6)
        seed = params.get("seed", 42)

        madnis_state = SData.integrator_states.get("MadNIS", None)

        if not no_output:
            OUTPUT_DIR = verify_path("outputs")
            directory = Path(OUTPUT_DIR, Settings.settings['run_name'].replace(" ", "_"), subroutine)
            if not os.path.exists(str(directory)):
                os.makedirs(str(directory))
                shell_print(f"Created output folder at {directory}")
            shell_print(f"Output will be at {directory}")

        Settings.settings["layered_integrator"]["integrator_type"] = "madnis"
        Settings.settings["layered_integrator"]["pretrain_c_flow"] = False
        madnis_kwargs = Settings.get_integrator_kwargs()
        integrand_kwargs = Settings.get_integrand_kwargs()
        param_kwargs = Settings.get_parameterisation_kwargs()
        madnis_integrator: MadnisIntegrator = Integrator.from_settings(
            Settings.settings
        )
        integrand = madnis_integrator.integrand
        if madnis_state is None:
            shell_print(
                f"WARNING: Could not find MadNIS state at '{file}'. Will use untrained MadNIS instance.")
        else:
            madnis_integrator.import_state(madnis_state)
            shell_print("Successfully imported madnis state from file.")

        # Will hold integration results to write to text file and plot
        Data = SlicePlotData(graph_properties=integrand.graph_properties,
                             settings=Settings.settings,
                             madnis_kwargs=madnis_kwargs,
                             integrand_kwargs=integrand_kwargs,
                             param_kwargs=param_kwargs,
                             EPS=EPS,)

        res = None
        madnis_obs = SData.observables.get("MadNIS", None)
        match integrand.training_phase:
            case "real":
                tgt = integrand.target.real_central_value
                if tgt:
                    res = tgt
                elif madnis_obs is not None:
                    res = madnis_obs.real_central_value
            case "imag":
                tgt = integrand.target.imag_central_value
                if tgt:
                    res = tgt
                elif madnis_obs is not None:
                    res = madnis_obs.imag_central_value

        madnis: MadNIS = madnis_integrator.madnis
        rng = np.random.default_rng(seed)
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
            t = np.linspace(t_min, t_max, n_samples_1d) * (1 - EPS)
            continuous = origin[None, :] + direction[None, :] * t.reshape(-1, 1)
            # Get the slice data
            layer_input = madnis_integrator.init_layer_data(n_samples_1d)
            layer_input.continuous = continuous
            layer_input.discrete = discrete
            acc: TrainingAccumulator = integrand.eval_integrand(layer_input, "training")
            func_val = acc.training_data.training_result[0].ravel()
            if res:
                func_val /= res
            # Molest MadNIS to get the probabilities for the slice
            x_all = torch.as_tensor(
                np.hstack([discrete, continuous]),
                device=madnis_integrator.device)
            prob = madnis.flow.prob(x_all).numpy(force=True).ravel()
            Data.slices1d.append(Slice(t=t, func_val=func_val, prob=prob))

        max_batch_size = 10_000  # To avoid memory issues when evaluating the flow.
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
            acc: TrainingAccumulator = integrand.eval_integrand(layer_input, "training")
            func_val = acc.training_data.training_result[0].reshape(n_samples_2d, n_samples_2d)
            if res:
                func_val /= res
            # Molest MadNIS to get the probabilities for the slice
            x_all = torch.as_tensor(
                np.hstack([discrete, continuous]),
                device=madnis.dummy.device,
                dtype=madnis.dummy.dtype)
            prob = np.empty((n_samples_2d**2,), dtype=integrand.dtype)

            n_eval = 0
            while n_eval < n_samples_2d**2:
                n = min(max_batch_size, n_samples_2d**2 - n_eval)
                with torch.no_grad():
                    prob[n_eval:n_eval+n] = madnis.flow.prob(x_all[n_eval:n_eval+n, :]).numpy(force=True).reshape(-1)
                n_eval += n
            prob = prob.reshape(n_samples_2d, n_samples_2d)
            Data.slices2d.append(Slice(t=grid_2d, func_val=func_val, prob=prob))

        # IMPORTANT: close the worker functions, or your script will hang
        integrand.end()

        if no_output:
            quit()

        run_name = Data.settings['run_name'].replace(' ', '_')
        filename = run_name + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+".pkl"
        file = Path(directory, filename)
        with file.open("wb") as f:
            torch.save(Data, f)

        if not no_plot:
            plot_slices(file, comment)

        return Data

    except KeyboardInterrupt:
        shell_print(f"\nCaught KeyboardInterrupt — stopping workers: {e}")
        integrand.end()
    except Exception as e:
        shell_print(f"\nCaught Exception — stopping workers: {e}")
        integrand.end()
    finally:
        integrand.end()


def plot_slices(file: str, comment: str = "") -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    from torch import load

    file: Path = verify_path(file)
    with file.open('rb') as f:
        Data: SlicePlotData = load(f, weights_only=False)

    shell_print(f"Plotting data from '{file}'")

    directory = file.parent
    filename = file.stem
    EPS = Data.EPS

    for i, slice in enumerate(Data.slices1d):
        slice: Slice
        fig, axs = plt.subplots(3, 1, sharex=True, layout="constrained",
                                height_ratios=[1, 0.3, 0.3], figsize=(6, 8))
        axs: List[plt.Axes]
        axs[0].plot(slice.t, np.abs(slice.func_val), label="|I / <I>|")
        axs[0].plot(slice.t, slice.prob, label="Probability")
        axs[0].legend()
        axs[1].plot(slice.t, np.abs(slice.func_val) / np.abs(slice.prob))
        axs[2].plot(slice.t, np.sign(slice.func_val))  # * np.sign(slice.prob))
        # axs[0].set_ylabel("|Integrand Value|")
        axs[1].set_ylabel("|Ratio|")
        axs[2].set_ylabel("sgn(I)")
        axs[0].set_yscale("log")
        axs[1].set_yscale("log")
        axs[2].set_yticks([-1, 0, 1])
        axs[2].set_yticklabels(['-', '0', '+'])
        fig.suptitle(f"1D Slices #{i} for {Data.settings['run_name']}")
        plt.savefig(
            Path(directory, filename + f"_slice1d_{i}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    cmap = plt.get_cmap('plasma')
    colors_low = cmap(np.linspace(0, 0.2, 128))
    colors_high = cmap(np.linspace(0.3, 1, 128))
    all_colors = np.vstack((colors_low, colors_high))
    cmap_segmented = colors.LinearSegmentedColormap.from_list('plasma', all_colors)
    fraction = 0.046  # Default fraction for colorbar size
    padding = 0.04  # Default padding between plot and colorbar
    threshold = 4  # Threshold for switching to TwoSlopeNorm

    for i, slice in enumerate(Data.slices2d):
        slice: Slice
        data_log_titles = [r"|I / <I>|", "Probability", r"|Ratio|"]
        data_log = [np.abs(slice.func_val), slice.prob, np.abs(slice.func_val) / slice.prob]
        data_discrete = np.sign(slice.func_val).astype(np.float64)  # * np.sign(slice.prob)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8),
                                 sharex=True, sharey=True, constrained_layout=True)
        (ax1, ax2), (ax3, ax4) = axes

        # Plot the funcval and probability with shared log color scale
        # vmin = min(d[d > 0].min() for d in data_log)
        # vmax = max(d.max() for d in data_log)
        # log_range = min(d[d > 0].min() for d in data_log) / max(d.max() for d in data_log)

        imgs = []
        for ax, data, title in zip([ax1, ax2, ax3], data_log, data_log_titles):
            data: np.ndarray = np.log10(data, out=np.full_like(data, np.nan, dtype=np.float64), where=(data > 0))
            vmin = data.min(where=(~np.isnan(data)), initial=np.inf)
            vmax = data.max(where=(~np.isnan(data)), initial=-np.inf)
            log_range = vmax - vmin
            log_norm = (
                colors.Normalize(vmin=vmin, vmax=vmax) if log_range < threshold
                else colors.TwoSlopeNorm(vcenter=vmax - threshold/2, vmin=vmin, vmax=vmax)
            )
            log_cm = cmap if log_range < threshold else cmap_segmented
            ax: plt.Axes
            im = ax.imshow(data, norm=log_norm, cmap=log_cm,
                           origin='lower', extent=[0, 1, 0, 1])
            imgs.append(im)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=fraction, pad=padding)

        # ratio = np.abs(slice.func_val) / np.abs(slice.prob)
        # log_norm_ratio = colors.LogNorm(vmin=ratio[ratio > 0].min(), vmax=ratio.max())
        # imratio = ax3.imshow(np.abs(slice.func_val) / np.abs(slice.prob), norm=log_norm_ratio, cmap='plasma',
        #                      origin='lower', extent=[0, 1, 0, 1])
        # ax3.set_title("|Ratio|")

        # Define Normalization for the 4th (Discrete: -1, 0, 1)
        # Boundaries are set at the midpoints to center the colors
        discrete_cmap = colors.ListedColormap(['#e74c3c', '#ecf0f1', '#2ecc71'])  # Red, Grey, Green, italian
        bounds = [-1.5, -0.5, 0.5, 1.5]
        discrete_norm = colors.BoundaryNorm(bounds, discrete_cmap.N)
        ax4: plt.Axes
        im4 = ax4.imshow(data_discrete, cmap=discrete_cmap, norm=discrete_norm,
                         origin='lower', extent=[0, 1, 0, 1])
        ax4.set_title("sgn(I)")

        # Add Colorbars
        # Shared colorbar for the first two, will be placed to the right of ax2
        # cbar_f_p = fig.colorbar(imgs[0], ax=[ax1, ax2], fraction=fraction, pad=padding)

        # Dedicated colorbar for the ratio
        # cbar_ratio = fig.colorbar(imratio, ax=ax3, fraction=fraction, pad=padding)

        # Dedicated colorbar for the discrete plot
        cbar_disc = fig.colorbar(im4, ax=ax4, fraction=fraction, pad=padding)
        # cbar_disc.set_label('Sign')
        cbar_disc.set_ticks([-1, 0, 1])
        cbar_disc.set_ticklabels(['-', '0', '+'])
        fig.suptitle(f"2D Slices #{i} for {Data.settings['run_name']}")

        plt.savefig(
            Path(directory, filename + f"_slice2d_{i}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
