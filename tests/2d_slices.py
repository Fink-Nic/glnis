# type: ignore
from numpy.typing import NDArray
from typing import List
from dataclasses import dataclass
from pathlib import Path
from glnis.utils.helpers import verify_path


class SlicePlotData:
    def __init__(self,
                 slices1d=[],
                 slices2d=[],
                 EPS: float = 1e-6
                 ) -> None:
        self.slices1d: List[Slice] = slices1d
        self.slices2d: List[Slice] = slices2d
        self.EPS = EPS


@dataclass
class Slice:
    origin: NDArray
    dirs: List[NDArray]
    func_val: NDArray
    prob: NDArray


def run_slice_plots() -> SlicePlotData | None:

    import numpy as np
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # Slice parameters
    rng = np.random.default_rng(10)
    EPS = 1e-6
    n_slices = 3
    n_cont = 7
    n_samples = 100
    t_max = np.sqrt(n_cont)
    sigma = 1/t_max

    # Will hold integration results to write to text file and plot
    Data = SlicePlotData(EPS=EPS,)

    for i in range(n_slices):
        # Draw samples and make the dirs perpendicular
        v1, v2, origin = [rng.uniform(size=(n_cont)) for _ in range(3)]
        v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
        v1 = v1 - v2.dot(v2)*v2
        dir1 = v1 / np.linalg.norm(v1)
        dir2 = v2
        # Find the coordinate origin
        # Assume the plane is parameterized by t1*dir1 + t2*dir2 + origin
        # dir1 = (a1, a2, ..., an), dir2 = (b1, b2, ..., bn), origin = (x1, x2, ..., xn)
        # Want to find solutions to the system of equations:
        # min(xk + t1*ak + t2*bk)=0 for k=1,...,n
        # max(xk + t1*ak + t2*bk)=1 for k=1,...,n
        # This will give us
        # t1_max = np.max((1 - origin) / dir1)
        # t1_min = -np.max(origin / dir2)
        # t2_max = np.max((1 - origin) / dir2)
        # t2_min = -np.max(origin / dir2)
        # t_max = max(t1_max, t2_max)
        # t_min = min(t1_min, t2_min)
        t_max = np.max((1 - origin) / (dir1 + dir2))
        t_min = -np.max(origin / (dir1 + dir2))
        origin = origin + t_min * dir1 + t_min * dir2
        dir1 = dir1 * (t_max - t_min)
        dir2 = dir2 * (t_max - t_min)
        # Now the square with the edge points
        # origin,
        # origin + dir1,
        # origin + dir2,
        # origin + dir1 + dir2
        # is completely outside the hypercube for each coordinate component
        # We now want to shrink the square until it touches the hypercube
        t1_max = -np.inf
        t1_min = np.inf
        t2_max = -np.inf
        t2_min = np.inf
        # TODO: find the smallest t1_max, t2_max, and largest t1_min, t2_min such that for all t and all components:
        # Can we calculate the distance d of a point p to the hypercube along a direction v?
        # Along just a single component k:
        # dk = (pk - 1) / vk, pk > 1 ~ t_max
        # dk = pk / vk, pk < 0 ~ t_min
        # And for the entire square? We need to find dk_max along all edges.
        # Along edge specified by ok + t1*dir1k:
        # in the interval t1 e [0, 1], we should have
        # dk_min|t1 = (ok + dir1k) / dir2k, dir1k > 0
        # dk_min|t1 = ok / dir2k, dir1k < 0
        # dk_max|t1 = ok / dir2k, dir1k > 0
        # dk_max|t1 = (ok + dir1k) / dir2k, dir1k < 0
        # origin + t1_max*dir1 + t*dir2 >= 1
        # origin + t*dir1 + t2_max*dir2 >= 1
        # origin + t1_min*dir1 + t*dir2 <= 0
        # origin + t*dir1 + t2_min*dir2 <= 0
        for k in range(n_cont):
            ok = origin[k]
            d1, d2 = dir1[k], dir2[k]
            # Update t1_max and t2_max
            t1_max_prospective = -np.inf
            t2_max_prospective = -np.inf
            t1_min_prospective = np.inf
            t2_min_prospective = np.inf
            if d1 != 0:
                t1_min_prospective = max(-ok / d1, (1 - ok) / d1) if d1 < 0 else min(-ok / d1, (1 - ok) / d1)
                t1_max_prospective = min(-ok / d1, (1 - ok) / d1) if d1 < 0 else max(-ok / d1, (1 - ok) / d1)
            if d2 != 0:
                t2_min_prospective = max(-ok / d2, (1 - ok) / d2) if d2 < 0 else min(-ok / d2, (1 - ok) / d2)
                t2_max_prospective = min(-ok / d2, (1 - ok) / d2) if d2 < 0 else max(-ok / d2, (1 - ok) / d2)

            # if t1_max_prospective > t1_min_prospective:
            if t1_max_prospective < 1:
                t1_max = max(t1_max, t1_max_prospective)
            if t1_min_prospective > 0:
                t1_min = min(t1_min, t1_min_prospective)
            # if t2_max_prospective > t2_min_prospective:
            if t2_max_prospective < 1:
                t2_max = max(t2_max, t2_max_prospective)
            if t2_min_prospective > 0:
                t2_min = min(t2_min, t2_min_prospective)
            print(f"Component {k}: t1_max_prospective={t1_max_prospective:.4f}, t1_min_prospective={t1_min_prospective:.4f}, t2_max_prospective={t2_max_prospective:.4f}, t2_min_prospective={t2_min_prospective:.4f}")
        t_min = min(t1_min, t2_min)
        t_max = max(t1_max, t2_max)

        s1max = (-origin.dot(dir1) + np.sum(dir1[dir1 > 0]))/dir1.dot(dir1)
        s1min = (-origin.dot(dir1) + np.sum(dir1[dir1 < 0]))/dir1.dot(dir1)
        s2max = (-origin.dot(dir2) + np.sum(dir2[dir2 > 0]))/dir2.dot(dir2)
        s2min = (-origin.dot(dir2) + np.sum(dir2[dir2 < 0]))/dir2.dot(dir2)
        try_something_new = True
        if try_something_new:
            s1max = np.max((-origin*dir1 + np.clip(dir1, 0, np.inf)) / dir1.dot(dir1))
            s1min = np.min((-origin*dir1 + np.clip(dir1, -np.inf, 0)) / dir1.dot(dir1))
            s2max = np.max((-origin*dir2 + np.clip(dir2, 0, np.inf)) / dir2.dot(dir2))
            s2min = np.min((-origin*dir2 + np.clip(dir2, -np.inf, 0)) / dir2.dot(dir2))
            t_min = min(s1min, s2min)
            t_max = max(s1max, s2max)
        print(f"Final t_min={t_min:.4f}, t_max={t_max:.4f}")
        origin = origin + t_min * dir1 + t_min * dir2
        dir1 = dir1 * (t_max - t_min)
        dir2 = dir2 * (t_max - t_min)

        t = np.linspace(0, 1, n_samples)
        t1, t2 = np.meshgrid(t, t)
        continuous = origin + t1[..., np.newaxis] * dir1 + t2[..., np.newaxis] * dir2
        inside_hcube_mask = np.all((continuous >= 0) & (continuous <= 1), axis=-1)

        func_val = np.where(inside_hcube_mask, np.exp(-np.sum(continuous**2, axis=-1) / (2*sigma**2)), np.nan)
        func_val = func_val.reshape(n_samples, n_samples)
        prob = func_val * 10.0**rng.uniform(-1.4, 1.4, size=(n_samples, n_samples))

        Data.slices2d.append(Slice(origin=origin, dirs=[dir1, dir2], func_val=func_val, prob=prob))

    plot_slices(Data)


def plot_slices(file: SlicePlotData) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    directory = verify_path("tests")
    filename = ""

    Data = file

    for i, slice in enumerate(Data.slices1d):
        slice: Slice
        t = np.linspace(Data.EPS, 1 - Data.EPS, len(slice.func_val))
        fig, axs = plt.subplots(3, 1, sharex=True, layout="constrained",
                                height_ratios=[1, 0.3, 0.3], figsize=(6, 8))
        axs: List[plt.Axes]
        axs[0].plot(t, np.abs(slice.func_val), label="|I / <I>|")
        axs[0].plot(t, slice.prob, label="Probability")
        axs[0].legend()
        axs[1].plot(t, np.abs(slice.func_val) / slice.prob)
        axs[2].plot(t, np.sign(slice.func_val))
        axs[1].set_ylabel("|Ratio|")
        axs[2].set_ylabel("sgn(I)")
        axs[0].set_yscale("log")
        axs[1].set_yscale("log")
        axs[2].set_xlim(0, 1)
        axs[2].set_yticks([-1, 0, 1])
        axs[2].set_yticklabels(['-', '0', '+'])
        fig.suptitle(f"1D Slices #{i} for")
        plt.savefig(
            Path(directory, filename + f"_slice1d_{i}.png"), dpi=300, bbox_inches="tight"
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

    for i, slice in enumerate(Data.slices2d):
        slice: Slice
        data_log_titles = ["|I / <I>|", "Probability"]
        data_log = [np.abs(slice.func_val), slice.prob]
        data_log = [d / np.nanmean(d) for d in data_log]  # Normalize by mean for better color scaling
        data_log = [np.log10(d, out=np.full_like(d, np.nan, dtype=np.float64), where=(d > 0)) for d in data_log]
        data_discrete = np.where(
            np.isfinite(slice.func_val),
            np.sign(slice.func_val),
            np.nan).astype(
            np.float64)  # * np.sign(slice.prob)

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
        data3 = np.abs(slice.func_val / slice.prob)
        # data3 /= np.nanmean(data3)  # Normalize by mean for better color scaling
        data3 = np.log10(data3, out=np.full_like(data3, np.nan, dtype=np.float64), where=(data3 > 0))
        im = ax3.imshow(data3, cmap=cmap3, extent=[0, 1, 0, 1],
                        norm=norm3, origin='lower')
        ax3.set_title("|Ratio|")
        cb3 = fig.colorbar(im, ax=ax3, fraction=fraction, pad=padding, extend='both')
        cb3.set_ticks(ticks=[-high_threshold, 0, high_threshold],
                      labels=[f"e{-high_threshold:+.0f}", "1", f"e{high_threshold:+.0f}"])

        discrete_cmap = colors.ListedColormap(['#e74c3c', 'lightgrey', '#2ecc71'])  # Red, Grey, Green, italian
        discrete_cmap.set_bad(color='lightgrey')  # Color for NaN values
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
        fig.suptitle(f"2D Slices #{i}")

        plt.savefig(
            Path(directory, filename + f"_slice2d_{i}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)


if __name__ == "__main__":
    run_slice_plots()
