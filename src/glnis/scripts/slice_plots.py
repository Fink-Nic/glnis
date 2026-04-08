# type: ignore
from numpy.typing import NDArray
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from glnis.utils.helpers import shell_print, verify_path
from glnis.core.accumulator import GraphProperties, TrainingAccumulator
from glnis.scripts.sampler_comparison import SamplerCompData


@dataclass
class LinSpace:
    start: float = 0.0
    stop: float = 1.0
    num: int = 1000


@dataclass
class Slice:
    origin: NDArray
    discrete: List[int]
    dirs: List[NDArray | int]
    grid: List[LinSpace]
    func_val: NDArray | None = None
    prob: Dict[str, NDArray] | None = None


class SlicePlotData:
    def __init__(self,
                 graph_properties: GraphProperties,
                 settings: Dict[str, Any] = dict(),
                 madnis_kwargs: Dict[str, Any] = dict(),
                 integrand_kwargs: Dict[str, Any] = dict(),
                 param_kwargs: Dict[str, Any] = dict(),
                 slices1d=None,
                 slices2d=None,
                 itg: float | None = None,
                 EPS: float = 1e-6
                 ) -> None:
        self.graph_properties = graph_properties
        self.settings: Dict[str, Any] = settings
        self.madnis_kwargs: Dict[str, Any] = madnis_kwargs
        self.integrand_kwargs: Dict[str, Any] = integrand_kwargs
        self.param_kwargs: Dict[str, Any] = param_kwargs
        self.slices1d: List[Slice] = slices1d or []
        self.slices2d: List[Slice] = slices2d or []
        self.EPS: float = EPS
        self.itg: float | None = itg


def run_slice_plots(
    file: str | SamplerCompData,
    settings_file: str = "",
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
    from glnis.core.integrand import MPIntegrand
    import traceback

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
        shell_print(f"Working on {file}")

    if len(SData.integrator_states) == 0:
        shell_print(f"No integrator states found in `SamplerCompData` object, exiting...")
        quit()

    Settings = SettingsParser(SData.settings)
    if settings_file:
        if isinstance(settings_file, str):
            shell_print(f"Overwriting settings with '{settings_file}'.")
        NewSettings = SettingsParser(settings_file)
        Settings.settings["scripts"]["slice_plots"] = NewSettings.settings.get(
            "scripts", dict()).get("slice_plots", dict())

    # Slice parameters
    scripts: Dict[str, Any] = Settings.settings.get("scripts", dict())
    params: Dict[str, Any] = scripts.get(subroutine, dict())
    slices: List[Dict[str, float | Dict[str, float]]] = params.get("slice", [])
    if not isinstance(slices, list):
        slices = [slices]
    if len(slices) == 0:
        shell_print(f"No slices defined in settings, exiting...")
    EPS = params.get("EPS", 1e-6)
    seed = params.get("seed", 42)
    rng = np.random.default_rng(seed)

    if force_directory is not None:
        directory = Path(force_directory)
    elif not no_output:
        OUTPUT_DIR = verify_path("outputs")
        directory = Path(OUTPUT_DIR, Settings.settings['run_name'].replace(" ", "_"), subroutine)
        if not os.path.exists(str(directory)):
            os.makedirs(str(directory))
            shell_print(f"Created output folder at {directory}")
        shell_print(f"Output will be at {directory}")

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
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
        # Get result to normalize the function values
        itg = None
        madnis_obs = SData.result.get("MadNIS", None)
        match integrand.training_phase:
            case "real":
                tgt = integrand.target.real_central_value
                if tgt:
                    itg = tgt
                elif madnis_obs is not None:
                    itg = madnis_obs.real_central_value
            case "imag":
                tgt = integrand.target.imag_central_value
                if tgt:
                    itg = tgt
                elif madnis_obs is not None:
                    itg = madnis_obs.imag_central_value

        if itg is not None and not only_plot:
            shell_print(f"Using integral estimate {itg:.4e} to normalize function values.")
        else:
            if not only_plot:
                shell_print(f"No integral estimate found. Function values will not be normalized.")

        # Initialize the integrators
        integrators: Dict[str, Integrator] = dict()

        def free_integrators():
            for integrator in integrators.values():
                integrator.free()

        for integrator_type, state in SData.integrator_states.items():
            itype = integrator_type.lower()
            match itype:
                case "madnis":
                    Settings.settings['layered_integrator']['integrator_type'] = "madnis"
                    integrators['madnis'] = MadnisIntegrator(integrand, **Settings.get_integrator_kwargs())
                    integrators['madnis'].import_state(state)
                    if not only_plot:
                        shell_print(f"Imported MadNIS state.")
                case "vegas":
                    Settings.settings['layered_integrator']['integrator_type'] = "vegas"
                    integrators['vegas'] = VegasIntegrator(integrand, **Settings.get_integrator_kwargs())
                    integrators['vegas'].import_state(state)
                    if not only_plot:
                        shell_print(f"Imported Vegas state.")

                case "havana":
                    Settings.settings['layered_integrator']['integrator_type'] = "havana"
                    integrators['havana'] = HavanaIntegrator(integrand, **Settings.get_integrator_kwargs())
                    integrators['havana'].import_state(state)
                    if not only_plot:
                        shell_print(f"Imported Havana state.")
                    # integrators['havana'].train(10, 100000)
                case "naive":
                    if not only_plot:
                        shell_print(f"No point in plotting slices for 'Naive' integrator. Skipping...")
                case _:
                    shell_print(f"Unrecognized integrator type '{integrator_type}' in file. Skipping...")

        # Will hold integration results to write to text file and plot
        Data = SlicePlotData(graph_properties=integrand.graph_properties,
                             settings=Settings.settings,
                             madnis_kwargs=Settings.get_integrator_kwargs(),
                             integrand_kwargs=integrand_kwargs,
                             param_kwargs=parameterisation_kwargs,
                             EPS=EPS,
                             itg=itg)

        for slice in slices:
            origin: List[float] | None = slice.get("origin", None)
            discrete: List[int] | None = slice.get("discrete", None)
            if origin is None:
                origin = rng.uniform(0.05, 0.95, size=integrand.continuous_dim)
            if discrete is None:
                discrete = [rng.integers(0, ddim) for ddim in integrand.discrete_dims]
            if not len(origin) == integrand.continuous_dim:
                shell_print(
                    f"Integrand has {integrand.continuous_dim} continuous dimensions, but origin has {len(origin)}. Skipping slice...")
                continue
            if not len(integrand.discrete_dims) == len(discrete):
                shell_print(
                    f"Integrand has {len(integrand.discrete_dims)} discrete dimensions, but {len(discrete)} were provided. Skipping slice...")
                continue
            dirs = slice.get("dirs", [])
            grid = slice.get("grid", [dict()])
            grid = [LinSpace(**ls) for ls in grid if isinstance(ls, dict)]
            if len(dirs) == 0:
                _dir = rng.uniform(EPS, 1, size=integrand.continuous_dim)
                dirs = [_dir / np.linalg.norm(_dir)]
            if len(grid) < len(dirs):
                grid += [LinSpace()] * (len(dirs) - len(grid))
            grid = grid[:len(dirs)]
            for d in dirs:
                if isinstance(d, int):
                    origin[d] = 0.0
            dirs = [[1 if i == d else 0 for i in range(integrand.continuous_dim)]
                    if isinstance(d, int) else d for d in dirs]
            for d in dirs:
                if len(d) != integrand.continuous_dim:
                    shell_print(
                        f"Integrand has {integrand.continuous_dim} continuous dimensions, but direction {d} has {len(d)}. Skipping slice...")
                    continue
            dirs = [np.array(d) for d in dirs]
            match len(dirs):
                case 1:
                    Data.slices1d.append(Slice(origin=np.array(origin), discrete=discrete, dirs=dirs, grid=grid))
                case 2:
                    Data.slices2d.append(Slice(origin=np.array(origin), discrete=discrete, dirs=dirs, grid=grid))
                case _:
                    shell_print(
                        f"Only 1D and 2D slices are supported, but {len(dirs)} were provided. Skipping slice...")
                    continue

        if len(Data.slices1d) == 0 and len(Data.slices2d) == 0:
            shell_print(f"No valid slices defined in settings, exiting...")
            quit()

        # Generate the grids
        slices1d: List[Tuple[Slice, Tuple[NDArray, NDArray]]] = []
        for s1d in Data.slices1d:
            grid = s1d.grid[0]
            n_samples_1d = grid.num
            # Sample a random discrete point
            if s1d.discrete:
                discrete = np.tile(np.array(s1d.discrete, dtype=np.uint64), (n_samples_1d, 1))
            else:
                discrete = np.empty((n_samples_1d, 0), dtype=np.uint64)
            # Sample a random direction in the continuous space
            origin = s1d.origin
            direction = s1d.dirs[0]
            grid.start = -np.min(np.divide(
                origin, direction, out=np.full_like(origin, np.inf),
                                 where=(direction != 0))) + EPS
            grid.end = np.min(np.divide(
                1 - origin, direction, out=np.full_like(origin, np.inf),
                where=(direction != 0))) - EPS
            grid1d = np.linspace(**asdict(grid))
            continuous = origin[None, :] + direction[None, :] * grid1d.reshape(-1, 1)
            slices1d.append((s1d, (discrete, continuous)))
            layer_input = LayerData(
                n_samples_1d,
                n_mom=3*graph_properties.n_loops,
                n_cont=integrand.continuous_dim,
                n_disc=len(integrand.discrete_dims),)
            layer_input.discrete, layer_input.continuous = discrete, continuous
            acc: TrainingAccumulator = integrand.eval_integrand(layer_input, "training")
            func_val = acc.training_data.training_result[0].ravel()
            if itg:
                func_val /= itg
            s1d.func_val = func_val
        if len(slices1d) > 0 and not only_plot:
            shell_print(f"Generated 1D grids and populated func_vals.")

        slices2d: List[Tuple[Slice, Tuple[NDArray, NDArray]]] = []
        for s2d in Data.slices2d:
            n_samples_2d = s2d.grid[0].num * s2d.grid[1].num
            # Sample a random discrete point
            if s2d.discrete:
                discrete = np.tile(np.array(s2d.discrete, dtype=np.uint64), (n_samples_2d, 1))
            else:
                discrete = np.empty((n_samples_2d, 0), dtype=np.uint64)
            # Sample a random point and create a grid in the 2 selected dimensions
            origin = s2d.origin
            grid_2d = np.stack(arrays=np.meshgrid(
                np.linspace(**asdict(s2d.grid[0])),
                np.linspace(**asdict(s2d.grid[1]))),
                axis=-1).reshape(-1, 2)
            continuous = origin + grid_2d[:, [0]] * s2d.dirs[0] + grid_2d[:, [1]] * s2d.dirs[1]
            inside_hcube_mask = np.all((continuous >= Data.EPS) & (continuous <= 1.0 - Data.EPS), axis=-1)
            layer_input = LayerData(
                int(np.sum(inside_hcube_mask)),
                n_mom=3*graph_properties.n_loops,
                n_cont=integrand.continuous_dim,
                n_disc=len(integrand.discrete_dims),)
            layer_input.continuous = continuous[inside_hcube_mask]
            layer_input.discrete = discrete[inside_hcube_mask]
            acc: TrainingAccumulator = integrand.eval_integrand(layer_input, "training")
            func_val = np.zeros(n_samples_2d)
            func_val[inside_hcube_mask] = acc.training_data.training_result[0].ravel()
            if itg:
                func_val /= itg
            s2d.func_val = func_val.reshape(s2d.grid[0].num, s2d.grid[1].num)
            slices2d.append((s2d, (discrete, continuous)))
        if len(slices2d) > 0 and not only_plot:
            shell_print(f"Generated 2D grids and populated func_vals.")

        for s1d, (discrete, continuous) in slices1d:
            integrator_probs = dict()
            for itype, integrator in integrators.items():
                prob: NDArray = integrator.probe_prob(discrete, continuous)
                integrator_probs[itype] = prob.ravel()
            s1d.prob = integrator_probs
        if len(slices1d) > 0 and not only_plot:
            shell_print(f"Populated probs for 1D slices.")

        for s2d, (discrete, continuous) in slices2d:
            integrator_probs = dict()
            for itype, integrator in integrators.items():
                prob: NDArray = integrator.probe_prob(discrete, continuous)
                integrator_probs[itype] = prob.reshape(s2d.func_val.shape)
            s2d.prob = integrator_probs
        if len(slices2d) > 0 and not only_plot:
            shell_print(f"Populated probs for 2D slices.")
        # IMPORTANT: close the worker functions, or your script will hang
        free_integrators()

        if no_output:
            quit()

        if only_plot:
            plot_slices(file=Data, comment=comment, force_directory=force_directory)
            return Data

        run_name = Data.settings['run_name'].replace(' ', '_')
        filename = run_name + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+".pkl"
        file = Path(directory, filename)
        with file.open("wb") as f:
            torch.save(Data, f)

        if not no_plot:
            plot_slices(file=file, comment=comment, force_directory=force_directory)

        return Data

    except KeyboardInterrupt:
        shell_print(f"\nCaught KeyboardInterrupt — stopping workers...")
        integrand.free()
        free_integrators()
    except Exception as e:
        shell_print(f"\nCaught Exception {e} — stopping workers...")
        integrand.free()
        free_integrators()
        print(traceback.format_exc())
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
        directory = verify_path("outputs")
        filename = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    else:
        file: Path = verify_path(file)
        with file.open('rb') as f:
            Data: SlicePlotData = load(f, weights_only=False)
        shell_print(f"Plotting data from '{file}'")

        directory = file.parent
        filename = file.stem

    if force_directory is not None:
        directory = verify_path(force_directory)

    precision = 3  # For table printout of numpy arrays
    cmap_name = 'plasma'
    use_blue_green_red = True  # Whether to use a custom blue-green-red colormap for the ratio plot
    fraction = 0.062  # Default fraction for colorbar size
    padding = -0.02  # Default padding between plot and colorbar
    high_threshold = 1
    center = -2
    low_threshold = -10
    low_high_ratio = 0.25  # Ratio of low to high colors in the segmented colormap
    low_high_gap = 0.05  # Gap between low and high colors in the segmented colormap

    for i, slice in enumerate(Data.slices1d):
        slice: Slice
        grid1d = np.linspace(**asdict(slice.grid[0]))
        for itype, prob in slice.prob.items():
            fig = plt.figure(layout="constrained", figsize=(6, 10))
            gs = fig.add_gridspec(4, 1, height_ratios=[1, 0.3, 0.3, 0.3])
            axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])]
            ax_table = fig.add_subplot(gs[3, 0])
            axs[1].sharex(axs[0])
            axs[2].sharex(axs[0])
            axs: List[plt.Axes]
            axs[0].plot(grid1d, np.abs(slice.func_val), label="|I / <I>|")
            axs[0].plot(grid1d, prob, label="Probability")
            axs[0].legend()
            axs[1].plot(grid1d, np.abs(slice.func_val) / prob)
            axs[2].plot(grid1d, np.sign(slice.func_val))
            axs[1].set_ylabel("|Ratio|")
            axs[2].set_ylabel("sgn(I)")
            axs[0].set_yscale("log")
            axs[1].set_yscale("log")
            axs[2].set_yticks([-1, 0, 1])
            axs[2].set_yticklabels(['-', '0', '+'])
            axs[2].set_xlabel("t")

            true_scale = np.nanmean(np.abs(slice.func_val) / prob)
            ax_table.axis('off')
            with np.printoptions(precision=precision, suppress=True):
                table_data = [
                    [r"$|<I/p|_{\mathrm{slice}}> / <I>|$", f"{true_scale:.3e}"],
                    [r"$\mathrm{origin, channel}$", f"{slice.origin}, {slice.discrete}"],
                    [r"$u$", f"{slice.dirs[0]}"],
                ]
            table = ax_table.table(cellText=table_data,
                                   cellLoc='center', loc='center', bbox=[0.2, 0.0, 0.6, 1.0])
            table.auto_set_font_size(False)
            table.set_fontsize(9)

            fig.suptitle(f"1D Slices #{i} for {Data.settings['run_name']} using {itype}")
            plt.savefig(
                Path(directory, filename + f"_slice1d_{itype}_{i}.png"), dpi=300, bbox_inches="tight"
            )
            plt.close(fig)

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

    data_log_titles = [r"$\frac{|I|}{<I>}$", r"$p$", r"$p \frac{<I>}{|I|}$"]

    for i, slice in enumerate(Data.slices2d):
        slice: Slice
        extent = [slice.grid[0].start, slice.grid[0].stop, slice.grid[1].start, slice.grid[1].stop]
        for itype, prob in slice.prob.items():
            data_log = [np.abs(slice.func_val), prob]
            data_log = [d / np.nanmean(d) for d in data_log]  # Normalize by mean for better color scaling
            data_log = [np.log10(d, out=np.full_like(d, np.nan, dtype=np.float64), where=(d > 0)) for d in data_log]
            data3 = data_log[1] - data_log[0]  # np.abs(slice.func_val / prob)
            true_scale = np.nanmean(np.abs(slice.func_val) / prob)
            data_discrete = np.sign(slice.func_val).astype(np.float64)

            fig = plt.figure(figsize=(11.5, 7.5), constrained_layout=True)
            fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)
            gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.30], wspace=0.02, hspace=0.03)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            axh1 = fig.add_subplot(gs[0, 2])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
            axh2 = fig.add_subplot(gs[1, 2])
            ax_table = fig.add_subplot(gs[2, :])
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
                               origin='lower', extent=extent)
                imgs.append(im)
                ax.set_title(f"Normalized {title}")
                cb12 = fig.colorbar(im, ax=ax, fraction=fraction, pad=padding, extend='both')
                cb12.set_ticks(ticks=[low_threshold, center, high_threshold],
                               labels=[f"e{low_threshold:+.0f}", f"e{center:+.0f}", f"e{high_threshold:+.0f}"])
            # data3 /= np.nanmean(data3)  # Normalize by mean for better color scaling
            # data3 = np.log10(data3, out=np.full_like(data3, np.nan, dtype=np.float64), where=(data3 > 0))
            im = ax3.imshow(data3, cmap=cmap3, extent=extent,
                            norm=norm3, origin='lower')
            ax3.set_title(f"Oversampling {data_log_titles[-1]}")
            cb3 = fig.colorbar(im, ax=ax3, fraction=fraction, pad=padding, extend='both')
            cb3.set_ticks(ticks=[-high_threshold, 0, high_threshold],
                          labels=[f"e{-high_threshold:+.0f}", "1", f"e{high_threshold:+.0f}"])

            discrete_cmap = colors.ListedColormap(['#e74c3c', '#ecf0f1', '#2ecc71'])  # Red, Grey, Green, italian
            bounds = [-1.5, -0.5, 0.5, 1.5]
            discrete_norm = colors.BoundaryNorm(bounds, discrete_cmap.N)
            ax4: plt.Axes
            im4 = ax4.imshow(data_discrete, cmap=discrete_cmap, norm=discrete_norm,
                             origin='lower', extent=extent)
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
            axh1.set_title("Normalized Values")
            axh1.set_xticks(ticks=[low_threshold, center, high_threshold],
                            labels=[f"e{low_threshold:+.0f}", f"e{center:+.0f}", f"e{high_threshold:+.0f}"])
            axh2_inner_edges = np.linspace(-high_threshold, high_threshold, 31)
            axh2_outer_width = max(2*high_threshold * 0.1, 1e-3)
            axh2_bins = np.concatenate((
                [-high_threshold - axh2_outer_width],
                axh2_inner_edges,
                [high_threshold + axh2_outer_width],
            ))
            axh2.hist(data3.ravel(), bins=axh2_bins,
                      density=True, histtype='step', alpha=0.75, linewidth=1.5, color='purple')
            # axh2.legend()
            axh2.set_title(f"Weighted oversampling {data_log_titles[-1]}.")
            axh2.set_xticks(ticks=[-high_threshold, 0, high_threshold],
                            labels=[f"e{-high_threshold:+.0f}", "1", f"e{high_threshold:+.0f}"])

            ax_table.axis('off')
            with np.printoptions(precision=precision, suppress=True):
                table_data = [
                    [r"$|<I/p|_{\mathrm{slice}}> / <I>|$", f"{true_scale:.3e}"],
                    [r"$\mathrm{origin, channel}$", f"{slice.origin}, {slice.discrete}"],
                    [r"$u$", f"{slice.dirs[0]}"],
                    [r"$v$", f"{slice.dirs[1]}"],
                ]
            table = ax_table.table(cellText=table_data,
                                   cellLoc='center', loc='center', bbox=[0.0, 0.0, 1.0, 1.0])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            fig.suptitle(f"2D Slices #{i} for {Data.settings['run_name']} using {itype}")

            plt.savefig(
                Path(directory, filename + f"_slice2d_{itype}_{i}.png"), dpi=300, bbox_inches="tight"
            )
            plt.close(fig)
