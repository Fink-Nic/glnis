# type: ignore
import functools
import numpy as np
from numpy.typing import NDArray, DTypeLike
from abc import ABC, abstractmethod
from time import perf_counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Literal, Iterable, Any

from glnis.utils.helpers import Colour, error_fmter, chunks


type LayerResult = NDArray | List | None


@dataclass
class GraphProperties:
    edge_src_dst_vertices: List[List[int]]
    edge_masses: List[float]
    edge_momentum_shifts: List[List[float]]
    graph_external_vertices: List[int]
    graph_signature: List[List[int]]
    momtrop_edge_weight: List[float] = field(default_factory=list)
    lmb_array: NDArray = field(default_factory=list)
    edge_external_sigs: List[List[float]] = field(default_factory=list)
    external_momenta: List[List[float]] = field(default_factory=list)
    orientation_ids: List[int] = field(default_factory=list)
    orientation_signatures: List[List[int]] = field(default_factory=list)
    generation_channel_id: int = 0
    e_cm: float = 0.0

    def __post_init__(self: 'GraphProperties'):
        if len(self.orientation_ids) != len(self.orientation_signatures):
            raise ValueError("Length of orientation_ids and orientation_signatures must match.")
        TOLERANCE = 1E-10
        self.n_loops: int = len(self.graph_signature[0])
        self.n_edges: int = len(self.edge_masses)
        self.edge_ismassive: list[bool] = [
            mass > TOLERANCE for mass in self.edge_masses]
        self.lmb_array = np.array(self.lmb_array, dtype=np.uint64)
        self.n_channels = self.lmb_array.shape[0]
        self.n_orientations = len(self.orientation_ids)

        try:
            # Calculate the inverse lmb transforms, ordered as the LMBs in graph properties
            self.channel_transforms = np.array(
                self.graph_signature)[self.lmb_array].reshape(self.n_channels, self.n_loops, self.n_loops)
            # Inverse transforms of each channel
            self.channel_inv_transforms = np.linalg.inv(self.channel_transforms)
        except:
            self.channel_transforms = np.zeros((0, self.n_loops, self.n_loops), dtype=np.float64)
            self.channel_inv_transforms = np.zeros((0, self.n_loops, self.n_loops), dtype=np.float64)


@dataclass
class IntegrationResult:
    n_points: int = 0
    real_central_value: float = 0
    real_error: float = 0
    imag_central_value: float = 0
    imag_error: float = 0
    abs_real_central_value: float = 0
    abs_real_error: float = 0
    abs_imag_central_value: float = 0
    abs_imag_error: float = 0
    real_rsd: float = 0
    imag_rsd: float = 0
    abs_real_rsd: float = 0
    abs_imag_rsd: float = 0
    real_tvar: float = 0
    imag_tvar: float = 0
    abs_real_tvar: float = 0
    abs_imag_tvar: float = 0
    total_time: float = 0

    def __post_init__(self: 'IntegrationResult'):
        """Calculates derived observables such as RSD and time per sample."""
        time_per_sample = 0
        # No harm in recalculating these
        if self.n_points > 0:
            time_per_sample = self.total_time / self.n_points
            if self.real_central_value:
                self.real_rsd = self._rsd(self.real_central_value, self.real_error, self.n_points)
            if self.imag_central_value:
                self.imag_rsd = self._rsd(self.imag_central_value, self.imag_error, self.n_points)
            if self.abs_real_central_value:
                self.abs_real_rsd = self._rsd(
                    self.abs_real_central_value, self.real_error, self.n_points)
            if self.abs_imag_central_value:
                self.abs_imag_rsd = self._rsd(
                    self.abs_imag_central_value, self.imag_error, self.n_points)

        # Don't want to overwrite these in case total_time is not provided
        self.real_tvar = self.real_tvar or self.real_rsd**2 * time_per_sample
        self.imag_tvar = self.imag_tvar or self.imag_rsd**2 * time_per_sample
        self.abs_real_tvar = self.abs_real_tvar or self.abs_real_rsd**2 * time_per_sample
        self.abs_imag_tvar = self.abs_imag_tvar or self.abs_imag_rsd**2 * time_per_sample

    def combine_with(self: 'IntegrationResult', other: 'IntegrationResult') -> None:
        def cc(s_c: float, o_c: float) -> float:
            return self._combine_central_value(self.n_points, other.n_points, s_c, o_c)

        def ce(s_e: float, o_e: float) -> float:
            return self._combine_error(self.n_points, other.n_points, s_e, o_e)

        self.real_central_value = cc(self.real_central_value, other.real_central_value)
        self.imag_central_value = cc(self.imag_central_value, other.imag_central_value)
        self.abs_real_central_value = cc(self.abs_real_central_value, other.abs_real_central_value)
        self.abs_imag_central_value = cc(self.abs_imag_central_value, other.abs_imag_central_value)
        self.real_error = ce(self.real_error, other.real_error)
        self.imag_error = ce(self.imag_error, other.imag_error)
        self.abs_real_error = ce(self.abs_real_error, other.abs_real_error)
        self.abs_imag_error = ce(self.abs_imag_error, other.abs_imag_error)

        self.n_points += other.n_points
        self.total_time += other.total_time
        self.__post_init__()  # Recalculate derived observables

    def str_report(self: 'IntegrationResult') -> str:
        return f"""Real: {
            self.real_central_value: .5f}  ± {
            self.real_error: .5f}, Imag: {
            self.imag_central_value: .5f}  ± {
            self.imag_error: .5f} """

    @staticmethod
    def _rsd(mean: float, err: float, n_points: int) -> float:
        if mean == 0:
            return 0
        return err / np.abs(mean) * np.sqrt(n_points)

    @staticmethod
    def _combine_central_value(s_n: float, o_n: float, s_c: float, o_c: float) -> float:
        return (s_n * s_c + o_n * o_c) / (s_n + o_n)

    @staticmethod
    def _combine_error(s_n: float, o_n: float, s_e: float, o_e: float) -> float:
        return np.sqrt(s_n**2 * s_e**2 + o_n**2 * o_e**2) / (s_n + o_n)


class LayerData:
    """
    What this should do:
        Pretty much everything lol
    """
    POSITIONS: Dict[str, int] = dict(
        jac=0,
        wgt=1,
        f_real=2,
        f_imag=3,
        momenta=4,
        continuous=5,
        discrete=6,
    )

    def __init__(self: 'LayerData',
                 n_points: int = 0,
                 n_mom: int = 0,
                 n_cont: int = 0,
                 n_disc: int = 0,
                 _existing_data: NDArray | None = None,
                 _existing_structure: NDArray | None = None,
                 _existing_active_structure: NDArray | None = None,
                 dtype: DTypeLike | None = None,):
        self._timestamp = perf_counter()
        self._t_init = perf_counter()

        self._pending_data: Dict[str, NDArray] = dict()
        self._structure = np.zeros(len(LayerData.POSITIONS), dtype=np.uint32)

        # Initializing from existing LayerData object data
        if not (_existing_data is None or _existing_structure is None):
            self._data = _existing_data.copy()
            self.dtype = self._data.dtype
            self._structure = _existing_structure.copy()
            if _existing_active_structure is None:
                self._active_structure = self._structure.copy()
            else:
                self._active_structure = _existing_active_structure.copy()
            self.n_points = self._data.shape[0]
            self.success = np.isfinite(self._data).all(axis=1)
        else:
            self.n_points = int(n_points)
            self.dtype = np.dtype(np.float64) if dtype is None else dtype
            # Set the dimensions of the data
            self._structure[LayerData.POSITIONS['jac']] = 1
            self._structure[LayerData.POSITIONS['wgt']] = 1
            self._structure[LayerData.POSITIONS['f_real']] = 1
            self._structure[LayerData.POSITIONS['f_imag']] = 1
            self._active_structure = self._structure.copy()
            self._structure[LayerData.POSITIONS['momenta']] = n_mom
            self._structure[LayerData.POSITIONS['continuous']] = n_cont
            self._structure[LayerData.POSITIONS['discrete']] = n_disc
            self._data = np.zeros(
                (self.n_points, self._structure.sum()), dtype=self.dtype)
            self._data[:, LayerData.POSITIONS['jac']] = 1
            self._data[:, LayerData.POSITIONS['wgt']] = 1
            self.success = np.ones(self.n_points, dtype=np.bool)

        self.failures: Dict[str, NDArray] = dict()
        self._processing_times = dict(
            processing=perf_counter() - self._timestamp)
        self._timestamp = perf_counter()

    def _to_layer_data(self: 'LayerData', value: LayerResult) -> NDArray:
        """
        Converts input to numpy NDArray of type self.dtype. Complex values must be passed
        separately as real and complex parts. Used in property setters.

        :param value: User data
        :type value: Tensor | NDArray | None
        :return:
        :rtype: NDArray
        """
        if value is None:
            return np.zeros(dtype=self.dtype, shape=(self.n_points, 0))

        match value:
            case list():
                output = np.array(value, dtype=self.dtype)
            case np.ndarray():
                output = value.astype(self.dtype)
            case _:
                raise ValueError(
                    "LayerData objects accept only numpy ndarray, list or None.")

        return output.reshape(self.n_points, -1)

    def _update_processing_times(self: 'LayerData', identifier: str = 'unspecified') -> None:
        processing_time = perf_counter() - self._timestamp
        if identifier in self._processing_times:
            self._processing_times[identifier] += processing_time
        else:
            self._processing_times[identifier] = processing_time
        self._timestamp = perf_counter()

    @staticmethod
    def _timer(func):
        @functools.wraps(func)
        def wrapper_timer(self: 'LayerData', *args, **kwargs):
            start_time = perf_counter()
            value = func(self, *args, **kwargs)
            self._processing_times['processing'] += perf_counter() - start_time
            return value

        return wrapper_timer

    @_timer
    def _set_data(self: 'LayerData',
                  name: str,
                  value: LayerResult,) -> None:
        # state will only be updated when update is called
        self._pending_data[name] = self._to_layer_data(value)

    @_timer
    def _get_data(self: 'LayerData', name: str):
        if name in self._pending_data.keys():
            self.update(name+'_getter')

        idx = LayerData.POSITIONS[name]
        offset = self._structure[:idx].sum()
        dim = self._active_structure[idx]
        return self._data[:, offset:offset+dim]

    @_timer
    def update(self, identifier: str = 'unspecified') -> None:
        """
        Updates state according to data in _pending_data. Will update the processing time of the
        processing step 'identifier' based on the elapsed time since last update. Will also mask
        non-finite values and update failures accordingly.
        """
        if len(self._pending_data) == 0:
            self._update_processing_times(identifier)
            return

        next_success = np.ones(self.n_points, dtype=np.bool)
        for v in self._pending_data.values():
            next_success = np.logical_and(
                next_success, np.isfinite(v).all(axis=1))

        new_failures_mask = np.logical_and(
            self.success, ~next_success)
        self.success = next_success

        if new_failures_mask.any():
            caused_failures = self._data[new_failures_mask]
            if identifier in self.failures:
                self.failures[identifier+"_"] = caused_failures
            else:
                self.failures[identifier] = caused_failures

        # Converting from iter to list allows us to remove items during loop
        for name in list(self._pending_data.keys()):
            idx = LayerData.POSITIONS[name]
            offset = self._structure[:idx].sum()
            value = self._pending_data.pop(name)
            dim = value.shape[1]
            self._active_structure[idx] = dim
            self._data[:, offset:offset+dim] = value

        # Update processing times according to time since last update
        self._update_processing_times(identifier)

    def get_processing_times(self) -> Dict[str, float]:
        """
        :return: Copy of the processing_times dict
        :rtype: Dict[str, float]
        """
        if len(self._pending_data) > 0:
            self.update('processing_times_getter')
        processing_times = dict(self._processing_times)
        processing_times['total_time'] = sum(t for t in processing_times.values())
        return processing_times

    def wake(self) -> None:
        """
        Updates the timestamp to the current time without any other side-effects.
        Useful when the LayerData object sits idle, e.g. after retrieving it from a Queue.
        """
        self._timestamp = perf_counter()

    @_timer
    def accumulate(self,
                   acc_type: Literal['default', 'training'] = 'default',
                   **kwargs) -> 'Accumulator':
        match acc_type:
            case 'default':
                return DefaultAccumulator(self, **kwargs)
            case 'training':
                return TrainingAccumulator(self, **kwargs)
            case _:
                raise NotImplementedError(
                    f"Accumulator of type {acc_type} not implemented for LayerData object.")

    @_timer
    def as_chunks(self, n_chunks: int, n_cores: int = 1) -> Iterable['LayerData']:
        if len(self._pending_data) > 0:
            self.update('as_chunks')

        data_chunks: Iterable[NDArray] = chunks(self._data, n_chunks)
        for chunk_id, data_chunk in enumerate(data_chunks):
            chunk = LayerData(
                _existing_data=data_chunk,
                _existing_structure=self._structure,
                _existing_active_structure=self._active_structure,
            )
            chunk._t_init = self._t_init
            # We add the metadata to the first chunk, and duplicate the processing time n_cores times
            if chunk_id == 0:
                chunk.failures = self.failures
            if chunk_id < n_cores:
                chunk._processing_times = self._processing_times

            yield chunk

    @property
    def jac(self: 'LayerData'):
        return self._get_data('jac')

    @jac.setter
    def jac(self: 'LayerData', value: LayerResult):
        self._set_data('jac', value)

    @property
    def wgt(self: 'LayerData'):
        return self._get_data('wgt')

    @wgt.setter
    def wgt(self: 'LayerData', value: LayerResult):
        self._set_data('wgt', value)

    @property
    def momenta(self: 'LayerData'):
        return self._get_data('momenta')

    @momenta.setter
    def momenta(self: 'LayerData', value: LayerResult):
        self._set_data('momenta', value)

    @property
    def f_real(self: 'LayerData'):
        return self._get_data('f_real')

    @f_real.setter
    def f_real(self: 'LayerData', value: LayerResult):
        self._set_data('f_real', value)

    @property
    def f_imag(self: 'LayerData'):
        return self._get_data('f_imag')

    @f_imag.setter
    def f_imag(self: 'LayerData', value: LayerResult):
        self._set_data('f_imag', value)

    @property
    def func_val(self: 'LayerData'):
        if len(self._pending_data) > 0:
            self.update('func_val_setter')
        return self._get_data('f_real') + 1j*self._get_data('f_imag')

    @func_val.setter
    def func_val(self: 'LayerData', value: NDArray[np.complexfloating]):
        if value is None:
            self._set_data('f_real', None)
            self._set_data('f_imag', None)
        if np.iscomplexobj(value):
            self._set_data('f_real', value.real)
            self._set_data('f_imag', value.imag)
        else:
            self._set_data('f_real', value)

    @property
    def continuous(self: 'LayerData'):
        return self._get_data('continuous')

    @continuous.setter
    def continuous(self: 'LayerData', value: LayerResult):
        self._set_data('continuous', value)

    @property
    def discrete(self: 'LayerData'):
        return self._get_data('discrete').astype(np.uint64)

    @discrete.setter
    def discrete(self: 'LayerData', value: LayerResult):
        self._set_data('discrete', value)


class AccumulatorModule(ABC):
    """
    Calculates a number of observables for a given LayerData object. Able to combine its
    data with a AccumulatorModule of the same type.
    """

    @abstractmethod
    def combine_with(self: 'AccumulatorModule', other: 'AccumulatorModule') -> None:
        pass

    @abstractmethod
    def str_report(self) -> str:
        return str(self.get_observables())

    @abstractmethod
    def get_observables(self) -> Dict[str, Any]:
        return dict()

    def finalise(self) -> None:
        """
        Finalises the state of the AccumulatorModule, e.g. by calculating derived observables.
        Called after all combinations are done, before reporting.
        """
        pass


class Accumulator:
    def __init__(self,
                 modules: List[AccumulatorModule],):
        self.modules = modules

    def combine_with(self: 'Accumulator', other: 'Accumulator') -> None:
        if not (type(self) is type(other) and len(self.modules) == len(other.modules)):
            raise TypeError("Cannot combine Accumulators of different types.")

        for s_module, o_module in zip(self.modules, other.modules):
            s_module.combine_with(o_module)

    def str_report(self) -> str:
        line = 125*"="
        report = [line]
        for module in self.modules:
            report.append(module.str_report())
        report.append(line)

        return "\n".join(report)

    def get_observables(self) -> Dict[str, Any]:
        obs = dict()
        for module in self.modules:
            obs.update(module.get_observables())
        return obs

    def finalise(self) -> None:
        for module in self.modules:
            module.finalise()


class DefaultAccumulator(Accumulator):
    def __init__(self,
                 data: LayerData | None = None,
                 target: IntegrationResult | None = None,
                 **kwargs):
        self.statistics = IntegrationStatistics(data, target=target)
        self.max_weight = MaxWeight(data)
        self.failures_monitor = FailuresMonitor(data)
        self.processing_times = ProcessingTimes(data)

        modules = [
            self.statistics,
            self.max_weight,
            self.failures_monitor,
            self.processing_times,
        ]
        super().__init__(modules)


class TrainingAccumulator(DefaultAccumulator):
    def __init__(self,
                 data: LayerData,
                 training_phase: Literal['real', 'imag', 'abs'] = 'real',
                 **kwargs):
        super().__init__(data=data, **kwargs)
        self.training_data = TrainingData(data, training_phase=training_phase)
        self.modules.append(self.training_data)
        self.training_mean = 0.0
        self.training_err = 0.0
        self.training_rsd = 0.0
        self.training_abs_rsd = 0.0
        self.training_tvar = 0.0
        self.training_abs_tvar = 0.0

    def _derive_training_observables(self) -> None:
        n_points = self.training_data.training_result.shape[0]
        time_per_sample = self.statistics.result.total_time / n_points
        sqrtn = np.sqrt(n_points)
        self.training_mean = self.training_data.training_result.mean()
        self.training_err = self.training_data.training_result.std() / sqrtn
        self.training_rsd = abs(self.training_err / self.training_mean) * sqrtn if self.training_mean != 0 else 0
        abs_res = np.abs(self.training_data.training_result)
        abs_res_mean = abs_res.mean()
        self.training_abs_rsd = abs(abs_res.std() / abs_res_mean) if abs_res_mean != 0 else 0
        self.training_tvar = self.training_rsd**2 * time_per_sample
        self.training_abs_tvar = self.training_abs_rsd**2 * time_per_sample

    def finalise(self) -> None:
        super().finalise()
        self._derive_training_observables()


class IntegrationStatistics(AccumulatorModule):
    """
    Implements AccumulatorModule. Calculates the integration result.

    :var n_points:
    """

    def __init__(self,
                 data: LayerData | None = None,
                 target: IntegrationResult | None = None,
                 precision: int = 2,):
        self.dtype = np.dtype(np.float64)
        self.target = target if target is not None else IntegrationResult()
        self.precision = precision
        if data is None:
            self.result = IntegrationResult()
        else:
            self.dtype = data.dtype
            m = data.success
            total_wgt = data.jac[m]*data.wgt[m]*data.func_val[m]
            abs_real_total_wgt = np.abs(total_wgt.real)
            abs_imag_total_wgt = np.abs(total_wgt.imag)
            total_time = data.get_processing_times().get('total_time', 0)
            self.result = IntegrationResult(
                n_points=data.n_points,
                real_central_value=total_wgt.real.mean(),
                real_error=total_wgt.real.std() / np.sqrt(data.n_points),
                imag_central_value=total_wgt.imag.mean(),
                imag_error=total_wgt.imag.std() / np.sqrt(data.n_points),
                abs_real_central_value=abs_real_total_wgt.mean(),
                abs_real_error=abs_real_total_wgt.std() / np.sqrt(data.n_points),
                abs_imag_central_value=abs_imag_total_wgt.mean(),
                abs_imag_error=abs_imag_total_wgt.std() / np.sqrt(data.n_points),
                total_time=total_time,
            )

    def combine_with(self: 'IntegrationStatistics', other: 'IntegrationStatistics') -> None:
        self.result.combine_with(other.result)

    def str_report(self) -> str:
        report = []
        report.append(
            f"Integration Result after {Colour.GREEN}{self.result.n_points}{Colour.END} evaluations:")
        if self.result.real_error > 0:
            report.append(f"""    {Colour.DARKCYAN}RE{Colour.END} : {
                error_fmter(self.result.real_central_value, self.result.real_error, self.precision)}""")
            err_rel = abs(self.result.real_error / self.result.real_central_value)
            err_col = Colour.GREEN if err_rel < 0.01 else Colour.RED
            rsd = self.result.real_rsd
            tvar = self.result.real_tvar
            atvar = self.result.abs_real_tvar
            report[-1] += f" ({err_col}{err_rel:.3%}{Colour.END})"
            if rsd > 0:
                report[-1] += f", RSD={rsd:.3f}"
            if tvar > 0:
                report[-1] += f", TVAR={tvar:.3e}"
            if atvar > 0:
                report[-1] += f", ATVAR={atvar:.3e}"
            if self.target.real_central_value != 0:
                diff = self.result.real_central_value - self.target.real_central_value
                rel_diff = abs(diff / self.target.real_central_value)
                if self.target.real_error > 0:
                    target_str = error_fmter(self.target.real_central_value, self.target.real_error, self.precision)
                else:
                    target_str = f"{self.target.real_central_value:<+25.16e}"

                report.append(
                    f"    vs target : {target_str} Δ = {diff:<+.{self.precision}e}")
                diff_col = Colour.GREEN if rel_diff < 0.01 else Colour.RED
                report[-1] += f" ({diff_col}{rel_diff:.3%}{Colour.END})"
                rsd = self.target.real_rsd
                tvar = self.target.real_tvar
                atvar = self.target.abs_real_tvar
                if rsd > 0:
                    report[-1] += f", RSD={rsd:.3f}"
                if tvar > 0:
                    report[-1] += f", TVAR={tvar:.3e}"
                if atvar > 0:
                    report[-1] += f", ATVAR={atvar:.3e}"

        if not self.result.imag_error <= 0:
            report.append(f"    {Colour.DARKCYAN}IM{Colour.END} : {
                error_fmter(self.result.imag_central_value, self.result.imag_error, self.precision)}")
            err_rel = abs(self.result.imag_error / self.result.imag_central_value)
            err_col = Colour.GREEN if err_rel < 0.01 else Colour.RED
            rsd = self.result.imag_rsd
            tvar = self.result.imag_tvar
            atvar = self.result.abs_imag_tvar
            report[-1] += f" ({err_col}{err_rel:.3%}{Colour.END})"
            if rsd > 0:
                report[-1] += f", RSD={rsd:.3f}"
            if tvar > 0:
                report[-1] += f", TVAR={tvar:.3e}"
            if atvar > 0:
                report[-1] += f", ATVAR={atvar:.3e}"
            if self.target.imag_central_value != 0:
                diff = self.result.imag_central_value - self.target.imag_central_value
                rel_diff = abs(diff / self.target.imag_central_value)
                if self.target.imag_error > 0:
                    target_str = error_fmter(self.target.imag_central_value, self.target.imag_error, self.precision)
                else:
                    target_str = f"{self.target.imag_central_value:<+25.16e}"

                report.append(
                    f"    vs target : {target_str} Δ = {diff:<+.{self.precision}e}")
                diff_col = Colour.GREEN if rel_diff < 0.01 else Colour.RED
                report[-1] += f" ({diff_col}{rel_diff:.3%}{Colour.END})"
                rsd = self.target.imag_rsd
                tvar = self.target.imag_tvar
                atvar = self.target.abs_imag_tvar
                if rsd > 0:
                    report[-1] += f", RSD={rsd:.3f}"
                if tvar > 0:
                    report[-1] += f", TVAR={tvar:.3e}"
                if atvar > 0:
                    report[-1] += f", ATVAR={atvar:.3e}"

        return "\n".join(line for line in report)

    def get_observables(self) -> Dict[str, Any]:
        return asdict(self.result).copy()


class MaxWeight(AccumulatorModule):
    def __init__(self,
                 data: LayerData | None = None,
                 dtype: DTypeLike | None = None,):
        self.dtype = np.dtype(np.float64) if dtype is None else dtype
        if data is None:
            self.real_pos_max_wgt: float = 0.0
            self.real_neg_max_wgt: float = 0.0
            self.imag_pos_max_wgt: float = 0.0
            self.imag_neg_max_wgt: float = 0.0
            self.real_pos_max_wgt_point = None
            self.real_neg_max_wgt_point = None
            self.imag_pos_max_wgt_point = None
            self.imag_neg_max_wgt_point = None
        else:
            self.dtype = data.dtype
            m = data.success
            total_wgt = data.jac[m]*data.wgt[m]*data.func_val[m]
            cont = data.continuous[m]
            disc = data.discrete[m]
            real_pos_idx = total_wgt.real.argmax()
            self.real_pos_max_wgt = total_wgt.real[real_pos_idx][0]
            self.real_pos_max_wgt_point = (disc[real_pos_idx], cont[real_pos_idx])
            real_neg_idx = total_wgt.real.argmin()
            self.real_neg_max_wgt = total_wgt.real[real_neg_idx][0]
            self.real_neg_max_wgt_point = (disc[real_neg_idx], cont[real_neg_idx])
            imag_pos_idx = total_wgt.imag.argmax()
            self.imag_pos_max_wgt = total_wgt.imag[imag_pos_idx][0]
            self.imag_pos_max_wgt_point = (disc[imag_pos_idx], cont[imag_pos_idx])
            imag_neg_idx = total_wgt.imag.argmin()
            self.imag_neg_max_wgt = total_wgt.imag[imag_neg_idx][0]
            self.imag_neg_max_wgt_point = (disc[imag_neg_idx], cont[imag_neg_idx])

    def combine_with(self: 'MaxWeight', other: 'MaxWeight') -> None:
        if other.real_pos_max_wgt_point is not None:
            if other.real_pos_max_wgt > self.real_pos_max_wgt:
                self.real_pos_max_wgt = other.real_pos_max_wgt
                self.real_pos_max_wgt_point = other.real_pos_max_wgt_point
        if other.real_neg_max_wgt_point is not None:
            if other.real_neg_max_wgt < self.real_neg_max_wgt:
                self.real_neg_max_wgt = other.real_neg_max_wgt
                self.real_neg_max_wgt_point = other.real_neg_max_wgt_point
        if other.imag_pos_max_wgt_point is not None:
            if other.imag_pos_max_wgt > self.imag_pos_max_wgt:
                self.imag_pos_max_wgt = other.imag_pos_max_wgt
                self.imag_pos_max_wgt_point = other.imag_pos_max_wgt_point
        if other.imag_neg_max_wgt_point is not None:
            if other.imag_neg_max_wgt < self.imag_neg_max_wgt:
                self.imag_neg_max_wgt = other.imag_neg_max_wgt
                self.imag_neg_max_wgt_point = other.imag_neg_max_wgt_point

    def str_report(self) -> str:
        report = ["Max weights and their location in x-space:"]

        def mwp_to_discstr_cont(mwp):
            disc, cont = mwp
            if disc.size > 0:
                disc_str = f"ch=[{' '.join(f"{d:.0f}" for d in disc)}] "
            else:
                disc_str = ""
            return disc_str, cont

        if self.real_pos_max_wgt > 0.:
            disc_str, cont = mwp_to_discstr_cont(self.real_pos_max_wgt_point)
            report.append(
                f"    {Colour.DARKCYAN}RE[+]{Colour.END} : {self.real_pos_max_wgt:<15.12e} at {disc_str}x={cont}")
        if self.real_neg_max_wgt < 0.:
            disc_str, cont = mwp_to_discstr_cont(self.real_neg_max_wgt_point)
            report.append(
                f"    {Colour.DARKCYAN}RE[-]{Colour.END} : {self.real_neg_max_wgt:<15.12e} at {disc_str}x={cont}")
        if self.imag_pos_max_wgt > 0.:
            disc_str, cont = mwp_to_discstr_cont(self.imag_pos_max_wgt_point)
            report.append(
                f"    {Colour.DARKCYAN}IM[+]{Colour.END} : {self.imag_pos_max_wgt:<15.12e} at {disc_str}x={cont}")
        if self.imag_neg_max_wgt < 0.:
            disc_str, cont = mwp_to_discstr_cont(self.imag_neg_max_wgt_point)
            report.append(
                f"    {Colour.DARKCYAN}IM[-]{Colour.END} : {self.imag_neg_max_wgt:<15.12e} at {disc_str}x={cont}")
        return "\n".join(line for line in report)

    def get_observables(self) -> Dict[str, Any]:
        return dict(
            real_pos_max_wgt=self.real_pos_max_wgt,
            real_neg_max_wgt=self.real_neg_max_wgt,
            imag_pos_max_wgt=self.imag_pos_max_wgt,
            imag_neg_max_wgt=self.imag_neg_max_wgt,
            real_pos_max_wgt_point=self.real_pos_max_wgt_point,
            real_neg_max_wgt_point=self.real_neg_max_wgt_point,
            imag_pos_max_wgt_point=self.imag_pos_max_wgt_point,
            imag_neg_max_wgt_point=self.imag_neg_max_wgt_point,
        )


class ProcessingTimes(AccumulatorModule):
    def __init__(self,
                 data: LayerData | None = None,
                 dtype: DTypeLike | None = None,
                 detailed: bool = False):
        self.dtype = np.dtype(np.float64) if dtype is None else dtype
        if data is None:
            self.processing_times = dict()
            self.n_points = 0
            self._t_init = perf_counter()
        else:
            self.processing_times = data.get_processing_times()
            self.n_points = data.n_points
            self._t_init = data._t_init
        self.time_processing = 0.0
        self.time_sampler = 0.0
        self.time_integrand = 0.0
        self.time_param = 0.0
        self.time_total = 0.0
        self.detailed = detailed

    def combine_with(self: 'ProcessingTimes', other: 'ProcessingTimes') -> None:
        self.n_points += other.n_points
        for identifier, time in other.processing_times.items():
            if identifier in self.processing_times:
                self.processing_times[identifier] += time
            else:
                self.processing_times[identifier] = time

    def str_report(self) -> str:
        mus_factor = 1.0e6 / self.n_points
        time_since_init = perf_counter() - self._t_init
        report = [
            f"Total elapsed time: {time_since_init:.2f} s | {self.time_total:.2f} CPU-s | {self.time_total*mus_factor:.1f} µs / eval. Breakdown by subroutine:"]
        subroutine_times = []
        processing_times = dict(
            sampler=self.time_sampler,
            param=self.time_param,
            integrand=self.time_integrand,
            processing=self.time_processing
        ) if not self.detailed else self.processing_times
        for identifier, time in processing_times.items():
            if identifier == 'total_time':
                continue
            perc_time = 100. * time / self.time_total
            if perc_time > 50.:
                subroutine_times.append(
                    f"{identifier}: {Colour.RED}{mus_factor*time:.1f}{Colour.END} µs ({Colour.RED}{int(perc_time)}%{Colour.END})")
            else:
                subroutine_times.append(
                    f"{identifier}: {mus_factor*time:.1f} µs ({int(perc_time)}%)")
        report.append(" | ".join(subroutine_times))

        return "\n".join(line for line in report)

    def get_observables(self) -> Dict[str, Any]:
        return dict(
            processing_times=self.processing_times,
            time_total=self.time_total,
            time_integrand=self.time_integrand,
            time_sampler=self.time_sampler,
            time_param=self.time_param,
            time_processing=self.time_processing,
        )

    def finalise(self) -> None:
        self.time_total = self.processing_times.get('total_time', 0)
        self.time_processing = self.processing_times.get('processing', 0)
        sampler_keys = [k for k in self.processing_times.keys() if 'sampler' in k]
        integrand_keys = [k for k in self.processing_times.keys() if 'integrand' in k]
        param_keys = [k for k in self.processing_times.keys() if 'param' in k]
        self.time_sampler = sum(self.processing_times.get(k, 0) for k in sampler_keys)
        self.time_integrand = sum(self.processing_times.get(k, 0) for k in integrand_keys)
        self.time_param = sum(self.processing_times.get(k, 0) for k in param_keys)


class FailuresMonitor(AccumulatorModule):
    def __init__(self,
                 data: LayerData | None = None,
                 dtype: DTypeLike | None = None,
                 max_display: int = 10):
        self.dtype = np.dtype(np.float64) if dtype is None else dtype
        self.max_display = max_display
        if data is None:
            self.n_points = 0
            self.n_success = 0
            self.failures = dict()
        else:
            self.n_points = data.n_points
            self.n_success = data.success.sum()
            self.failures = data.failures
            self.structure = np.zeros(len(LayerData.POSITIONS))

    def combine_with(self: 'FailuresMonitor', other: 'FailuresMonitor') -> None:
        self.n_points += other.n_points
        self.n_success += other.n_success
        for identifier, failures in other.failures.items():
            if identifier in self.failures:
                self.failures[identifier] = np.vstack(
                    [self.failures[identifier], failures], dtype=self.dtype)
            else:
                self.failures[identifier] = failures

    def str_report(self) -> str:
        if self.n_points == self.n_success:
            return f"""{Colour.GREEN}No Failures, successfully evaluated all {
                self.n_points}/{self.n_points} points!{Colour.END}"""

        n_failures = self.n_points - self.n_success
        report = [
            f"{Colour.RED}WARNING: Failed {n_failures} points!{Colour.END}"]
        n_displayed = 0
        for identifier, failures in self.failures.items():
            n_curr_id = min(failures.shape[0], self.max_display-n_displayed)
            for idx in range(n_curr_id):
                report.append(f"At '{identifier}': {failures[idx]}")
            n_displayed += n_curr_id
            if n_displayed >= self.max_display:
                break
        if n_failures > self.max_display:
            report.append(
                f"... And {Colour.RED}{n_failures-self.max_display}{Colour.END} more failed evaluations.")

        return "\n".join(line for line in report)

    def get_observables(self) -> Dict[str, Any]:
        return dict(
            n_points=self.n_points,
            n_success=self.n_success,
            failures=self.failures,
        )


class TrainingData(AccumulatorModule):
    FAILUREVALUE = 0.

    def __init__(self,
                 data: LayerData,
                 training_phase: Literal['real', 'imag', 'abs'] = 'real',):
        self.dtype = data.dtype
        self.training_phase = training_phase
        self.training_result: list[NDArray] | NDArray = []
        self.has_failures = len(data.failures) > 0
        if self.has_failures:
            data._data[~data.success] = self.FAILUREVALUE
        int_result = data.jac*data.wgt
        match self.training_phase:
            case 'real':
                int_result *= data.f_real
            case 'imag':
                int_result *= data.f_imag
            case 'abs':
                int_result *= np.abs(data.func_val)
            case _:
                raise ValueError(
                    "Training phase must be one of 'real', 'imag' or 'abs'.")
        int_result[np.isnan(int_result)] = self.FAILUREVALUE
        self.training_result.append(int_result)

    def combine_with(self: 'TrainingData', other: 'TrainingData') -> None:
        if not self.training_phase == other.training_phase:
            raise RuntimeError(
                "Cannot combine TrainingData objects with different phase.")
        self.has_failures = self.has_failures or other.has_failures
        self.training_result += other.training_result

    def str_report(self) -> str:
        return f"Trained on phase: {Colour.GREEN}{self.training_phase.upper()}{Colour.END}"

    def get_observables(self) -> Dict[str, Any]:
        return dict(
            training_phase=self.training_phase,
        )

    def finalise(self) -> None:
        self.training_result = np.vstack(self.training_result)
