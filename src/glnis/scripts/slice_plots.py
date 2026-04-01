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
                 slices1d=dict(),
                 slices2d=dict(),
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
    func_val: NDArray
    prob: Dict[str, NDArray]


def run_slice_plots(
    file: str | SamplerCompData,
    comment: str = "",
    no_output: bool = False,
    no_plot: bool = False,
    only_plot: bool = False,
    force_directory: str | None = None,
    subroutine: str = "slice_plots",
) -> SlicePlotData | None:

    import os
    import signal
    import numpy as np
    import torch

    from glnis.core.accumulator import LayerData
    from glnis.core.integrator import (
        Integrator,
        MadnisIntegrator,
        VegasIntegrator,
        HavanaIntegrator,
    )
    from glnis.core.parser import SettingsParser
    from madnis.integrator import Integrator as MadNIS
    from glnis.core.integrand import MPIntegrand

    if isinstance(file, SamplerCompData):
        SData = file
    else:
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

    if len(SData.integrator_states) == 0:
        shell_print(f"No integrator states found in `SamplerCompData` object at '{file}', exiting...")
        quit()

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        shell_print(f"Working on {file}")
        Settings = SettingsParser(SData.settings)

        # Slice parameters
        scripts: Dict[str, Any] = Settings.settings.get("scripts", dict())
        params: Dict[str, Any] = scripts.get("slice_plots", dict())
        n_samples_1d = params.get("n_samples_1d", 1000)
        n_samples_2d = params.get("n_samples_2d", 1000)
        n_slices_1d = params.get("n_slices_1d", 2)
        slice_dims_2d = params.get("slice_dims_2d", [[0, 1]])
        EPS = params.get("EPS", 1e-6)
        seed = params.get("seed", 42)
        rng = np.random.default_rng(seed)
        max_batch_size = 10_000  # To avoid memory issues when evaluating the flow.

        if force_directory is not None:
            directory = Path(force_directory)
        elif not no_output:
            OUTPUT_DIR = verify_path("outputs")
            directory = Path(OUTPUT_DIR, Settings.settings['run_name'].replace(" ", "_"), subroutine)
            if not os.path.exists(str(directory)):
                os.makedirs(str(directory))
                shell_print(f"Created output folder at {directory}")
            shell_print(f"Output will be at {directory}")

        # Initialize the integrand to be shared across integrators
        graph_properties = Settings.get_graph_properties()
        parameterisation_kwargs = Settings.get_parameterisation_kwargs()
        integrand_kwargs = Settings.get_integrand_kwargs()
        n_cores = integrand_kwargs.pop("n_cores", 16)
        verbose = integrand_kwargs.pop("verbose", False)
        integrand = MPIntegrand(
            graph_properties=graph_properties,
            param_kwargs=parameterisation_kwargs,
            integrand_kwargs=integrand_kwargs,
            n_cores=n_cores,
            verbose=verbose,
        )
        # Will hold integration results to write to text file and plot
        Data = SlicePlotData(graph_properties=integrand.graph_properties,
                             settings=Settings.settings,
                             madnis_kwargs=Settings.get_integrator_kwargs(),
                             integrand_kwargs=integrand_kwargs,
                             param_kwargs=parameterisation_kwargs,
                             EPS=EPS,)

        # Get result to normalize the function values
        res = None
        madnis_obs = SData.result.get("MadNIS", None)
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

        # Generate the grids
        grids1d = []
        func_vals1d = []
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
            grids1d.append((discrete, continuous))
            layer_input = LayerData(
                n_samples_1d,
                n_mom=3*graph_properties.n_loops,
                n_cont=integrand.continuous_dim,
                n_disc=len(integrand.discrete_dims),)
            layer_input.discrete, layer_input.continuous = discrete, continuous
            acc: TrainingAccumulator = integrand.eval_integrand(layer_input, "training")
            func_val = acc.training_data.training_result[0].ravel()
            if res:
                func_val /= res
            func_vals1d.append(func_val)

        grids2d = []
        func_vals2d = []
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
            grids2d.append((discrete, continuous))
            layer_input = LayerData(
                n_samples_2d**2,
                n_mom=3*graph_properties.n_loops,
                n_cont=integrand.continuous_dim,
                n_disc=len(integrand.discrete_dims),)
            layer_input.continuous = continuous
            layer_input.discrete = discrete
            acc: TrainingAccumulator = integrand.eval_integrand(layer_input, "training")
            func_val = acc.training_data.training_result[0].reshape(n_samples_2d, n_samples_2d)
            if res:
                func_val /= res
            func_vals2d.append(func_val)
        # Initialize the integrators

        integrators: Dict[str, Integrator] = dict()

        def free_integrators():
            for integrator in integrators.values():
                integrator.free()

        for integrator_type, state in SData.integrator_states.items():
            itype = integrator_type.lower()
            match itype:
                case "madnis":
                    if not isinstance(state, MadnisIntegrator.MadnisState):
                        shell_print(
                            f"Expected MadnisIntegrator state for 'MadNIS' integrator, but got {type(state)}. Skipping...")
                    Settings.settings['layered_integrator']['integrator_type'] = "madnis"
                    integrators['madnis'] = MadnisIntegrator(integrand, **Settings.get_integrator_kwargs())
                case "vegas":
                    if not isinstance(state, VegasIntegrator.VegasState):
                        shell_print(
                            f"Expected VegasIntegrator state for 'Vegas' integrator, but got {type(state)}. Skipping...")
                    Settings.settings['layered_integrator']['integrator_type'] = "vegas"
                    integrators['vegas'] = VegasIntegrator(integrand, **Settings.get_integrator_kwargs())
                case "havana":
                    if not isinstance(state, HavanaIntegrator.HavanaState):
                        shell_print(
                            f"Expected HavanaIntegrator state for 'Havana' integrator, but got {type(state)}. Skipping...")
                    Settings.settings['layered_integrator']['integrator_type'] = "havana"
                    integrators['havana'] = HavanaIntegrator(integrand, **Settings.get_integrator_kwargs())
                case "naive":
                    shell_print(f"No point in plotting slices for 'Naive' integrator. Skipping...")
                case _:
                    shell_print(f"Unrecognized integrator type '{integrator_type}' in file. Skipping...")

        for grid1d, func_vals in zip(grids1d, func_vals1d):
            discrete, continuous = grid1d
            integrator_probs = dict()
            for itype, integrator in integrators.items():
                prob = integrator.probe_prob(discrete, continuous)
                integrator_probs[itype] = prob.reshape(n_samples_1d)
            Data.slices1d.append(Slice(func_val=func_vals, prob=integrator_probs))

        for grid2d, func_vals in zip(grids2d, func_vals2d):
            discrete, continuous = grid2d
            integrator_probs = dict()
            for itype, integrator in integrators.items():
                prob = integrator.probe_prob(discrete, continuous)
                integrator_probs[itype] = prob.reshape(n_samples_2d, n_samples_2d)
            Data.slices2d.append(Slice(func_val=func_vals, prob=integrator_probs))
        # IMPORTANT: close the worker functions, or your script will hang
        free_integrators()

        if only_plot:
            plot_slices(file, comment, force_directory=force_directory)
            quit()

        if no_output:
            quit()

        run_name = Data.settings['run_name'].replace(' ', '_')
        filename = run_name + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+".pkl"
        file = Path(directory, filename)
        with file.open("wb") as f:
            torch.save(Data, f)

        if not no_plot:
            plot_slices(file, comment, force_directory=force_directory)

        return Data

    except KeyboardInterrupt:
        shell_print(f"\nCaught KeyboardInterrupt — stopping workers: {e}")
        integrand.free()
        free_integrators()
    except Exception as e:
        shell_print(f"\nCaught Exception — stopping workers: {e}")
        integrand.free()
        free_integrators()
    finally:
        integrand.free()
        free_integrators()


def plot_slices(file: str, comment: str = "", force_directory: str | None = None) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    from torch import load

    if isinstance(file, SlicePlotData):
        Data = file
        shell_print(f"Plottin SlicePlotData from argument")
    else:
        file: Path = verify_path(file)
        with file.open('rb') as f:
            Data: SlicePlotData = load(f, weights_only=False)
        shell_print(f"Plotting data from '{file}'")

        directory = file.parent
        filename = file.stem

    if force_directory is not None:
        directory = Path(force_directory)
        filename = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    EPS = Data.EPS

    for i, slice in enumerate(Data.slices1d):
        slice: Slice
        for itype, prob in slice.prob.items():
            fig, axs = plt.subplots(3, 1, sharex=True, layout="constrained",
                                    height_ratios=[1, 0.3, 0.3], figsize=(6, 8))
            axs: List[plt.Axes]
            axs[0].plot(slice.t, np.abs(slice.func_val), label="|I / <I>|")
            axs[0].plot(slice.t, prob, label="Probability")
            axs[0].legend()
            axs[1].plot(slice.t, np.abs(slice.func_val) / prob)
            axs[2].plot(slice.t, np.sign(slice.func_val))
            axs[1].set_ylabel("|Ratio|")
            axs[2].set_ylabel("sgn(I)")
            axs[0].set_yscale("log")
            axs[1].set_yscale("log")
            axs[2].set_yticks([-1, 0, 1])
            axs[2].set_yticklabels(['-', '0', '+'])
            fig.suptitle(f"1D Slices #{i} for {Data.settings['run_name']}")
            plt.savefig(
                Path(directory, filename + f"_slice1d_{itype}_{i}.png"), dpi=300, bbox_inches="tight"
            )
            plt.close(fig)

    cmap_name = 'plasma'
    use_blue_green_red = True  # Whether to use a custom blue-green-red colormap for the ratio plot
    fraction = 0.046  # Default fraction for colorbar size
    padding = 0.04  # Default padding between plot and colorbar
    high_threshold = 1
    center = -2
    low_threshold = -10
    low_high_ratio = 0.25  # Ratio of low to high colors in the segmented colormap
    low_high_gap = 0.05  # Gap between low and high colors in the segmented colormap
    cmap = plt.get_cmap(cmap_name)
    colors_low = cmap(np.linspace(0, low_high_ratio, 128))
    colors_high = cmap(np.linspace(low_high_ratio+low_high_gap, 1, 128))
    all_colors = np.vstack((colors_low, colors_high))
    cmap_segmented = colors.LinearSegmentedColormap.from_list(cmap_name, all_colors)
    cmap_segmented.set_bad(color='lightgrey')  # Color for NaN values
    cmap_segmented.set_under(color='black')  # Color for values below vmin
    cmap_segmented.set_over(color='white')  # Color for values above vmax
    norm12 = colors.TwoSlopeNorm(vcenter=center, vmin=low_threshold, vmax=high_threshold)
    norm3 = colors.Normalize(vmin=-high_threshold, vmax=high_threshold)
    if use_blue_green_red:
        blue = "#2b83ba"
        green = "#22bc27"
        red = "#d72d30"
        cmap3 = colors.LinearSegmentedColormap.from_list("core_scaling", [blue, green, red])
        cmap3.set_bad(color='lightgrey')  # Color for NaN values
        cmap3.set_under(color=blue)  # Color for values below vmin
        cmap3.set_over(color=red)  # Color for values above vmax
    else:
        cmap3 = plt.get_cmap(cmap_name)
        cmap3.set_bad(color='lightgrey')  # Color for NaN values
        cmap3.set_under(color='black')  # Color for values below vmin
        cmap3.set_over(color='white')  # Color for values above vmax

    data_log_titles = ["|I / <I>|", "Probability"]

    for i, slice in enumerate(Data.slices2d):
        slice: Slice
        for itype, prob in slice.prob.items():
            data_log = [np.abs(slice.func_val), prob]
            data_log = [d / np.nanmean(d) for d in data_log]  # Normalize by mean for better color scaling
            data_log = [np.log10(d, out=np.full_like(d, np.nan, dtype=np.float64), where=(d > 0)) for d in data_log]
            data_discrete = np.sign(slice.func_val).astype(np.float64)

            fig, axes = plt.subplots(2, 3, figsize=(10, 6.2),
                                     sharex=False, sharey=False, constrained_layout=True)
            (ax1, ax2, axh1), (ax3, ax4, axh2) = axes
            ax2.sharex(ax1)
            ax2.sharey(ax1)
            ax3.sharex(ax1)
            ax3.sharey(ax1)
            ax4.sharex(ax1)
            ax4.sharey(ax1)
            axh1.set_box_aspect(1)
            axh2.set_box_aspect(1)

            imgs = []
            for ax, data, title in zip([ax1, ax2], data_log, data_log_titles):
                # vmin = data.min(where=(~np.isnan(data)), initial=np.inf)
                # vmax = data.max(where=(~np.isnan(data)), initial=-np.inf)
                ax: plt.Axes
                im = ax.imshow(data, norm=norm12, cmap=cmap_segmented,
                               origin='lower', extent=[0, 1, 0, 1])
                imgs.append(im)
                ax.set_title(title)
                cb12 = fig.colorbar(im, ax=ax, fraction=fraction, pad=padding, extend='both')
                cb12.set_ticks(ticks=[low_threshold, center, high_threshold],
                               labels=[f"e{low_threshold:+.0f}", f"e{center:+.0f}", f"e{high_threshold:+.0f}"])
            data3 = np.abs(slice.func_val / prob)
            # data3 /= np.nanmean(data3)  # Normalize by mean for better color scaling
            data3 = np.log10(data3, out=np.full_like(data3, np.nan, dtype=np.float64), where=(data3 > 0))
            im = ax3.imshow(data3, cmap=cmap3, extent=[0, 1, 0, 1],
                            norm=norm3, origin='lower')
            ax3.set_title("|Ratio|")
            cb3 = fig.colorbar(im, ax=ax3, fraction=fraction, pad=padding, extend='both')
            cb3.set_ticks(ticks=[-high_threshold, 0, high_threshold],
                          labels=[f"e{-high_threshold:+.0f}", "1", f"e{high_threshold:+.0f}"])

            discrete_cmap = colors.ListedColormap(['#e74c3c', '#ecf0f1', '#2ecc71'])  # Red, Grey, Green, italian
            bounds = [-1.5, -0.5, 0.5, 1.5]
            discrete_norm = colors.BoundaryNorm(bounds, discrete_cmap.N)
            ax4: plt.Axes
            im4 = ax4.imshow(data_discrete, cmap=discrete_cmap, norm=discrete_norm,
                             origin='lower', extent=[0, 1, 0, 1])
            ax4.set_title("sgn(I)")

            cbar_disc = fig.colorbar(im4, ax=ax4, fraction=fraction, pad=padding)
            # cbar_disc.set_label('Sign')
            cbar_disc.set_ticks(ticks=[-1, 0, 1], labels=['-', '0', '+'])

            # Simple histograms to visualize variance
            axh1: plt.Axes
            axh2: plt.Axes
            # Add one explicit underflow and overflow bin outside the xtick range.
            axh1_inner_edges = np.linspace(low_threshold, high_threshold, 31)
            axh1_outer_width = max((high_threshold - low_threshold) * 0.1, 1e-3)
            axh1_bins = np.concatenate((
                [low_threshold - axh1_outer_width],
                axh1_inner_edges,
                [high_threshold + axh1_outer_width],
            ))
            axh1.hist(data_log[0].ravel(), bins=axh1_bins, label=data_log_titles[0],
                      density=True, histtype='step', alpha=0.75, linewidth=1.5)
            axh1.hist(data_log[1].ravel(), bins=axh1_bins, label=data_log_titles[1],
                      density=True, histtype='step', alpha=0.75, linewidth=1.5)
            axh1.legend()
            axh1.set_title("Normalized Log Values")
            axh1.set_xticks(ticks=[low_threshold, center, high_threshold],
                            labels=[f"e{low_threshold:+.0f}", f"e{center:+.0f}", f"e{high_threshold:+.0f}"])
            axh2_inner_edges = np.linspace(-high_threshold, high_threshold, 31)
            axh2_outer_width = max(2*high_threshold * 0.1, 1e-3)
            axh2_bins = np.concatenate((
                [-high_threshold - axh2_outer_width],
                axh2_inner_edges,
                [high_threshold + axh2_outer_width],
            ))
            axh2.hist(data3.ravel(), bins=axh2_bins, label="|I / <I>| / Probability",
                      density=True, histtype='step', alpha=0.75, linewidth=1.5, color='purple')
            axh2.legend()
            axh2.set_title("Weighted Log Ratio")
            axh2.set_xticks(ticks=[-high_threshold, 0, high_threshold],
                            labels=[f"e{-high_threshold:+.0f}", "1", f"e{high_threshold:+.0f}"])
            fig.suptitle(f"2D Slices #{i} for {Data.settings['run_name']}")

            plt.savefig(
                Path(directory, filename + f"_slice2d_{itype}_{i}.png"), dpi=300, bbox_inches="tight"
            )
            plt.close(fig)
