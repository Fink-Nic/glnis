# type: ignore
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from pickle import dump, load

from glnis.utils.helpers import shell_print, verify_path
from glnis.core.accumulator import IntegrationResult
from glnis.scripts.sampler_comparison import run_sampler_comp, SamplerCompData


@dataclass
class OLDRunData:
    comp_name: str
    block_name: str
    settings: Dict[str, Any]
    madnis_info: Dict[str, Any]
    plottables: SamplerCompData.Plottables
    target: IntegrationResult
    observables: IntegrationResult
    run_time: float


@dataclass
class RunData:
    comp_name: str
    block_name: str
    settings: Dict[str, Any]
    madnis_info: Dict[str, Any]
    plottables: SamplerCompData.Plottables
    target: IntegrationResult
    observables: Dict[str, Any]
    run_time: float


class OLDHParamCompData:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.sorted_by_comp_and_name: Dict[str, Dict[str, RunData]] = dict()
        self.sorted_by_obs: Dict[str, List[Tuple[str, str, float]]] = dict()
        self._total_comparisons = 0

    def check_if_done(self, comp_name: str, block_name: str) -> bool:
        if comp_name in self.sorted_by_comp_and_name:
            if block_name in self.sorted_by_comp_and_name[comp_name]:
                return True
        return False

    def add_result(
            self, comp_name: str, block_name: str, additional_params: Dict[str, Any],
            result: SamplerCompData, save: bool = True) -> None:
        if self.check_if_done(comp_name, block_name):
            shell_print(f"Result for comparison '{comp_name}' and block '{block_name}' already exists, skipping...")
            return
        run_data = self.to_run_data(comp_name, block_name, additional_params, result)
        if comp_name not in self.sorted_by_comp_and_name:
            self.sorted_by_comp_and_name[comp_name] = dict()
        self.sorted_by_comp_and_name[comp_name][block_name] = run_data

        all_obs_dict = asdict(run_data.result)
        all_obs_dict['run_time'] = run_data.run_time
        all_obs_dict['discrete_params'] = run_data.madnis_info.get('discrete flow total parameters', 0)
        all_obs_dict['continuous_params'] = run_data.madnis_info.get('continuous flow total parameters', 0)
        all_obs_dict['total_params'] = run_data.madnis_info.get('flow total parameters', 0)

        for obs_name, value in all_obs_dict.items():
            if value != 0:
                if obs_name not in self.sorted_by_obs:
                    self.sorted_by_obs[obs_name] = [(comp_name, block_name, value)]
                else:
                    self.sorted_by_obs[obs_name].append((comp_name, block_name, value))
                    self.sorted_by_obs[obs_name].sort(key=lambda x: x[2])

        self._total_comparisons += 1
        if save:
            self.save()

    def to_run_data(
            self, comp_name: str, block_name: str, additional_params: Dict[str, Any],
            result: SamplerCompData) -> RunData:
        madnis_observables = list(result.observables.values())[0] if result.observables else IntegrationResult()
        return RunData(
            comp_name=comp_name,
            block_name=block_name,
            settings=result.settings,
            madnis_info=result.madnis_info,
            plottables=result.plottables,
            target=result.target,
            observables=madnis_observables,
            run_time=additional_params.get('run_time', 0.0)
        )

    def save(self, file: str | None = None) -> None:
        path = Path(file if file is not None else self.filename)
        temp_path = path.with_suffix(".tmp")
        with temp_path.open("wb") as f:
            dump(self, f)
        temp_path.replace(path.with_suffix(".pkl"))


class HParamCompData:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.sorted_by_comp_and_name: Dict[str, Dict[str, RunData]] = dict()
        self.sorted_by_obs: Dict[str, List[Tuple[str, str, float]]] = dict()
        self._total_comparisons = 0

    def check_if_done(self, comp_name: str, block_name: str) -> bool:
        if comp_name in self.sorted_by_comp_and_name:
            if block_name in self.sorted_by_comp_and_name[comp_name]:
                return True
        return False

    def add_result(
            self, comp_name: str, block_name: str, additional_params: Dict[str, Any],
            result: SamplerCompData, save: bool = True) -> None:
        if self.check_if_done(comp_name, block_name):
            shell_print(f"Result for comparison '{comp_name}' and block '{block_name}' already exists, skipping...")
            return
        run_data = self.to_run_data(comp_name, block_name, additional_params, result)
        if comp_name not in self.sorted_by_comp_and_name:
            self.sorted_by_comp_and_name[comp_name] = dict()
        self.sorted_by_comp_and_name[comp_name][block_name] = run_data

        all_obs_dict = dict(run_data.observables)
        all_obs_dict['run_time'] = run_data.run_time
        all_obs_dict['discrete_params'] = run_data.madnis_info.get('discrete flow total parameters', 0)
        all_obs_dict['continuous_params'] = run_data.madnis_info.get('continuous flow total parameters', 0)
        all_obs_dict['total_params'] = run_data.madnis_info.get('flow total parameters', 0)

        for obs_name, value in all_obs_dict.items():
            try:
                value = float(value)
            except (ValueError, TypeError):
                continue
            if value != 0:
                if obs_name not in self.sorted_by_obs:
                    self.sorted_by_obs[obs_name] = [(comp_name, block_name, value)]
                else:
                    self.sorted_by_obs[obs_name].append((comp_name, block_name, value))
                    self.sorted_by_obs[obs_name].sort(key=lambda x: x[2])

        self._total_comparisons += 1
        if save:
            self.save()

    def to_run_data(
            self, comp_name: str, block_name: str, additional_params: Dict[str, Any],
            result: SamplerCompData) -> RunData:
        madnis_observables = list(result.observables.values())[0] if result.observables else IntegrationResult()
        return RunData(
            comp_name=comp_name,
            block_name=block_name,
            settings=result.settings,
            madnis_info=result.madnis_info,
            plottables=result.plottables,
            target=result.target,
            observables=madnis_observables,
            run_time=additional_params.get('run_time', 0.0)
        )

    def save(self, file: str | None = None) -> None:
        path = Path(file if file is not None else self.filename)
        temp_path = path.with_suffix(".tmp")
        with temp_path.open("wb") as f:
            dump(self, f)
        temp_path.replace(path.with_suffix(".pkl"))


def run_hyperparam_comparison(
    file: str,
    recovery_file: str = "",
    comment: str = "",
    no_output: bool = False,
    no_plot: bool = False,
    subroutine: str = "hyperparam_comparison",
) -> SamplerCompData | None:

    if Path(file).suffix == ".pkl":
        plot_hyperparam_comparison(file, comment)
        quit()

    from time import perf_counter
    import gc
    from glnis.utils.helpers import _open_fd_count, _fd_limit

    fd_limit = _fd_limit()

    from glnis.core.parser import SettingsParser

    shell_print(f"Working on settings {file}")
    MasterSettings = SettingsParser(file)

    if recovery_file:
        recovery_file = verify_path(recovery_file, suffix=".pkl")
        with recovery_file.open('rb') as f:
            Data: HParamCompData = load(f)
            if not isinstance(Data, HParamCompData):
                raise ValueError(f"Expected a HParamCompData object in the recovery file, but got {type(Data)}")
        shell_print(f"Recovered data from '{recovery_file}' with {Data._total_comparisons} comparisons.")
    else:
        PROJECT_ROOT = Path(__file__).parents[3]
        OUTPUT_DIR = "outputs"
        directory = Path(PROJECT_ROOT, OUTPUT_DIR, MasterSettings.settings['run_name'].replace(" ", "_"), subroutine)
        if not directory.exists():
            directory.mkdir(parents=True)
            shell_print(f"Created output folder at {directory}")
        shell_print(f"Output will be at {directory}")
        filename = Path(directory, datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        Data = HParamCompData(filename=str(filename))

    hline = 120 * "="
    # Training parameters
    scripts: Dict[str, Any] = MasterSettings.settings.get("scripts", dict())
    params_list: Dict[str, Any] = scripts.get(subroutine, dict())
    if not isinstance(params_list, list):
        params_list: List[Dict[str, Any]] = [params_list]

    for params in params_list:
        blocks = params.get('blocks', None)
        block_names = params.get('block_names', [])
        comp_name = params.get('comparison_name', 'no_name_set')
        if blocks is None:
            shell_print(f"No blocks to run in comparison '{comp_name}', skipping...")
            continue

        if not isinstance(blocks, list):
            blocks = [[blocks]]
        if len(blocks) == 0:
            shell_print(f"No blocks to run in comparison '{comp_name}', skipping...")
            continue
        if not isinstance(blocks[0], list):
            blocks = [blocks]
        if not isinstance(block_names, list):
            block_names = [block_names]
        if not len(blocks) == len(block_names):
            block_names = [f"noname{i}" for i in range(len(blocks))]

        for block_name, templates in zip(block_names, blocks):
            block_name = str(block_name)
            if Data.check_if_done(comp_name, block_name):
                shell_print(f"Comparison '{comp_name}' and block '{block_name}' already done, skipping...")
                continue
            NewSettings = MasterSettings.settings_with_additional_templates(templates)
            fd_before = _open_fd_count()
            if fd_before is not None:
                limit_msg = f"/{fd_limit}" if fd_limit is not None else ""
                shell_print(f"FD diagnostic before run: {fd_before}{limit_msg}")
            shell_print(f"Starting comparison '{comp_name}' and block '{block_name}'")
            shell_print(hline)
            shell_print(hline)

            before = perf_counter()
            run_result: SamplerCompData = run_sampler_comp(
                file=NewSettings.settings,
                no_naive=True,
                no_vegas=True,
                no_havana=True,
                export_states=False,
                no_output=True,
                subroutine='hpcomp_training_run')
            run_time = perf_counter() - before
            fd_after_run = _open_fd_count()

            Data.add_result(
                comp_name, block_name,
                additional_params={'run_time': run_time},
                result=run_result, save=not no_output)

            gc.collect()
            fd_after_gc = _open_fd_count()

            shell_print(hline)
            shell_print(hline)
            shell_print(f"Finished comparison '{comp_name}' and block '{block_name}' in {run_time:.2f} seconds.")
            if fd_after_run is not None:
                delta = fd_after_run - (fd_before if fd_before is not None else fd_after_run)
                limit_msg = f"/{fd_limit}" if fd_limit is not None else ""
                shell_print(f"FD diagnostic after run: {fd_after_run}{limit_msg} (delta={delta:+d})")
            if fd_after_gc is not None:
                delta = fd_after_gc - (fd_before if fd_before is not None else fd_after_gc)
                limit_msg = f"/{fd_limit}" if fd_limit is not None else ""
                shell_print(f"FD diagnostic after gc.collect(): {fd_after_gc}{limit_msg} (delta={delta:+d})")

    if no_output:
        return Data

    Data.save()

    if not no_plot:
        plot_hyperparam_comparison(Data.filename, comment)

    return Data


def plot_hyperparam_comparison(file: str, comment: str = "") -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import numpy as np
    from numpy.typing import NDArray

    def _finite_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if np.isfinite(parsed):
            return parsed
        return None

    file: Path = verify_path(file, suffix=".pkl")
    with file.open('rb') as f:
        Data: HParamCompData = load(f)
        if not isinstance(Data, HParamCompData):
            raise ValueError(f"Expected a HParamCompData object in the file, but got {type(Data)}")

    shell_print(f"Plotting data from '{file}'")

    directory = file.parent
    filename = file.stem

    for comp_name, blocks in Data.sorted_by_comp_and_name.items():
        shell_print(f"Plotting comparison '{comp_name}'...")
        subdirectory = Path(directory, comp_name.replace(" ", "_"))

        target = list(blocks.values())[0].target
        n_blocks = len(blocks)
        block_ticks = [block_name for block_name in blocks.keys()]
        cmap = colors.LinearSegmentedColormap.from_list(
            "core_scaling",
            ["#2b83ba", "#5ab4ac", "#abdda4", "#fdae61", "#d7191c"],
        )
        cols = [
            colors.to_hex(cmap(t))
            for t in np.linspace(0.0, 1.0, n_blocks, endpoint=True)
        ]

        fig1, axs1 = plt.subplots(4, 1, sharex=True, layout="constrained",
                                  height_ratios=(3, 1, 1, 1), figsize=(10, 8))
        axs1: list[plt.Axes]
        axs1[0].set_ylabel("loss")
        axs1[1].set_ylabel("RSD")
        axs1[2].set_ylabel("TVAR")
        axs1[3].set_ylabel("ATVAR")
        axs1[3].set_xlabel("Training steps")
        for ax in axs1:
            ax.set_yscale("log")
        fig2, axs2 = plt.subplots(4, 2, sharex=True, layout="constrained",
                                  height_ratios=(3, 1, 1, 1), figsize=(10, 8))
        axs2: NDArray[plt.Axes]
        # Column labels
        axs2[0, 0].set_title("RE")
        axs2[0, 1].set_title("IM")
        # Row labels
        axs2[0, 0].set_ylabel("I(f)")
        axs2[1, 0].set_ylabel("RSD")
        axs2[2, 0].set_ylabel("TVAR")
        axs2[3, 0].set_ylabel("ATVAR")
        for i in range(2):
            axs2[3, i].set_xticks(range(n_blocks), block_ticks, rotation=45)
            axs2[1, i].set_yscale("log")
            axs2[2, i].set_yscale("log")
            axs2[3, i].set_yscale("log")

        tgt_line_len = n_blocks - 1
        if target.real_central_value:
            axs2[0, 0].hlines(target.real_central_value, 0, tgt_line_len, color='red')
            if target.real_error:
                axs2[0, 0].fill_between(
                    [0, tgt_line_len],
                    target.real_central_value - target.real_error,
                    target.real_central_value + target.real_error,
                    color='red', alpha=0.3
                )
        if target.real_rsd:
            axs2[1, 0].hlines(target.real_rsd, 0, tgt_line_len, color='red')
        if target.real_tvar:
            axs2[2, 0].hlines(target.real_tvar, 0, tgt_line_len, color='red')
        if target.abs_real_tvar:
            axs2[3, 0].hlines(target.abs_real_tvar, 0, tgt_line_len, color='red')

        if target.imag_central_value:
            axs2[0, 1].hlines(target.imag_central_value, 0, tgt_line_len, color='red')
            if target.imag_error:
                axs2[0, 1].fill_between(
                    [0, tgt_line_len],
                    target.imag_central_value - target.imag_error,
                    target.imag_central_value + target.imag_error,
                    color='red', alpha=0.3
                )
        if target.imag_rsd:
            axs2[1, 1].hlines(target.imag_rsd, 0, tgt_line_len, color='red')
        if target.imag_tvar:
            axs2[2, 1].hlines(target.imag_tvar, 0, tgt_line_len, color='red')
        if target.abs_imag_tvar:
            axs2[3, 1].hlines(target.abs_imag_tvar, 0, tgt_line_len, color='red')

        fig3, axs3 = plt.subplots(nrows=2, ncols=1, sharex=True, layout="constrained", figsize=(8, 6))
        axs3: List[plt.Axes]
        axs3[-1].set_xticks(range(n_blocks), block_ticks, rotation=45)
        cont_param_list = []
        disc_param_list = []
        if not subdirectory.exists():
            subdirectory.mkdir()
        for i, (block_name, run_data) in enumerate(blocks.items()):
            losses, steps_losses = np.array(run_data.plottables.losses), np.array(run_data.plottables.steps_losses)
            rsds, steps_snapshot = np.array(run_data.plottables.rsds), np.array(run_data.plottables.steps_snapshot)
            tvars, atvars = np.array(run_data.plottables.tvars), np.array(run_data.plottables.abs_tvars)

            # Training progression data
            if len(steps_losses) and len(steps_snapshot):
                axs1[0].plot(steps_losses, losses, label=block_name, color=cols[i])
                axs1[1].scatter(steps_snapshot, rsds, label=block_name, color=cols[i])
                axs1[2].scatter(steps_snapshot, tvars, label=block_name, color=cols[i])
                axs1[3].scatter(steps_snapshot, atvars, label=block_name, color=cols[i])

            # Final integration results
            obs = run_data.observables
            if obs['real_error'] > 0:
                axs2[0, 0].errorbar(i, obs['real_central_value'], yerr=obs['real_error'],
                                    marker='o', markersize=5, capsize=5, color='black')
                axs2[1, 0].scatter(i, obs['real_rsd'], color='black')
                axs2[2, 0].scatter(i, obs['real_tvar'], color='black')
                axs2[3, 0].scatter(i, obs['abs_real_tvar'], color='black')
            if obs['imag_error'] > 0:
                axs2[0, 1].errorbar(i, obs['imag_central_value'], yerr=obs['imag_error'],
                                    marker='o', markersize=5, capsize=5, color='black')
                axs2[1, 1].scatter(i, obs['imag_rsd'], color='black')
                axs2[2, 1].scatter(i, obs['imag_tvar'], color='black')
                axs2[3, 1].scatter(i, obs['abs_imag_tvar'], color='black')

            # Additional observables
            num_param_discrete = run_data.madnis_info.get('discrete flow total parameters', 0)
            num_param_continuous = run_data.madnis_info.get('continuous flow total parameters', 0)
            num_param_total = run_data.madnis_info.get('flow total parameters', 0)
            if not num_param_continuous:
                num_param_continuous = num_param_total - num_param_discrete
            cont_param_list.append(num_param_continuous)
            disc_param_list.append(num_param_discrete)
            axs3[1].scatter(i, run_data.run_time, color='black')

        axs3[0].bar(range(n_blocks), cont_param_list, label='Continuous',
                    color='white', edgecolor='black', hatch='//')
        axs3[0].bar(range(n_blocks), disc_param_list, bottom=cont_param_list, label='Discrete',
                    color='white', edgecolor='black', hatch='..')
        axs3[0].set_ylabel("Number of parameters")
        axs3[0].set_yscale("log")
        axs3[1].set_ylabel("Run time (s)")
        axs3[0].legend(loc='upper left')

        axs1[0].legend(loc='upper right')
        fig1.suptitle(f"MadNIS training progression for {comp_name}")
        fig1.savefig(
            Path(subdirectory, filename + "_training_prog.png"), dpi=300, bbox_inches="tight"
        )
        fig2.suptitle(f"Integration results for {comp_name}")
        fig2.savefig(
            Path(subdirectory, filename + "_integration_result.png"), dpi=300, bbox_inches="tight"
        )
        fig3.suptitle(f"Additional observables for {comp_name}")
        fig3.savefig(
            Path(subdirectory, filename + "_additional_observables.png"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        shell_print(f"Finished plotting comparison '{comp_name}'. Plots saved to '{subdirectory}'.")

    all_rsds = []
    for (comp, block, real_err) in Data.sorted_by_obs['real_error']:
        _rsd = None
        for (c, b, real_mean) in Data.sorted_by_obs['real_central_value']:
            if c == comp and b == block and real_mean != 0:
                _rsd = real_err / abs(real_mean)
        for (c, b, n_points) in Data.sorted_by_obs['n_points']:
            if c == comp and b == block and n_points > 0:
                if _rsd is not None:
                    _rsd *= np.sqrt(n_points)
        if _rsd is not None:
            all_rsds.append((comp, block, _rsd))
    all_rsds.sort(key=lambda x: x[2])
    Data.sorted_by_obs['real_rsd'] = all_rsds
    ignore = ["n_points", "real_central_value", "imag_central_value",
              "abs_real_central_value", "abs_imag_central_value"]
    n_show = 5
    # Overall observables comparison
    observables_to_plot = [(obs, Data.sorted_by_obs[obs]) for obs in Data.sorted_by_obs.keys() if obs not in ignore]
    n_obs = len(observables_to_plot)
    fig4, axs4 = plt.subplots(nrows=n_obs, ncols=2, layout="constrained", figsize=(10, 4*n_obs))
    axs4: List[plt.Axes]
    axs4[0, 0].set_title(f"Lowest")
    axs4[0, 1].set_title(f"Highest")
    for i, (obs_name, obs_tuples) in enumerate(observables_to_plot):
        # obs_tuple: List[(comp_name, block_name, value)]
        n_blocks = len(obs_tuples)
        n = min(n_show, n_blocks)
        obs_values = [o[-1] for o in obs_tuples]
        axs4[i, 0].set_ylabel(obs_name)
        axs4[i, 0].set_yscale("log")
        axs4[i, 1].set_yscale("log")
        axs4[i, 0].set_xticks(range(n), [f"{comp}\n{block}" for comp, block, _ in obs_tuples[:n]], rotation=45)
        axs4[i, 1].set_xticks(range(n), [f"{comp}\n{block}" for comp, block, _ in obs_tuples[-n:]], rotation=45)
        axs4[i, 0].scatter(range(n), obs_values[:n], color='black')
        axs4[i, 1].scatter(range(n), obs_values[-n:], color='black')
        axs4[i, 0].set_ylim(bottom=0.9*min(obs_values[:n]), top=1.1*max(obs_values[:n]))
        axs4[i, 1].set_ylim(bottom=0.9*min(obs_values[-n:]), top=1.1*max(obs_values[-n:]))

    fig4.suptitle(f"Overall observables comparison {comment}")
    fig4.savefig(
        Path(directory, filename + "_overall_observables_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig4)
    shell_print(f"Finished plotting overall observables comparison. Plot saved to '{directory}'.")
