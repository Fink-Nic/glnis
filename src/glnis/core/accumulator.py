# type: ignore
import functools
import numpy as np
from numpy.typing import NDArray, DTypeLike
from abc import ABC, abstractmethod
from time import perf_counter
from dataclasses import dataclass, field
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

    def __post_init__(self: 'GraphProperties'):
        TOLERANCE = 1E-10
        self.n_loops: int = len(self.graph_signature[0])
        self.n_edges: int = len(self.edge_masses)
        self.edge_ismassive: list[bool] = [
            mass > TOLERANCE for mass in self.edge_masses]
        self.lmb_array = np.array(self.lmb_array, dtype=np.uint64)
        self.n_channels = self.lmb_array.shape[0]

        # Calculate the inverse lmb transforms, ordered as the LMBs in graph properties
        self.channel_transforms = np.array(
            self.graph_signature)[self.lmb_array].reshape(self.n_channels, self.n_loops, self.n_loops)
        # Inverse transforms of each channel
        self.channel_inv_transforms = np.linalg.inv(self.channel_transforms)


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
                self.failures[identifier+"_1"] = caused_failures
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
        return dict(self._processing_times)

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
    def as_chunks(self, n_chunks: int) -> Iterable['LayerData']:
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
            # We add the metadata to the first chunk, as this is before any MP happens
            if chunk_id == 0:
                chunk._processing_times = self._processing_times
                chunk.failures = self.failures
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


class Accumulator(AccumulatorModule):
    def __init__(self,
                 modules: List[AccumulatorModule],):
        self.modules = modules

    def combine_with(self: 'Accumulator', other: 'Accumulator') -> None:
        if not len(self.modules) == len(other.modules):
            raise TypeError("Cannot combine Accumulators of different types.")

        for s_module, o_module in zip(self.modules, other.modules):
            s_module.combine_with(o_module)

    def str_report(self) -> str:
        return "\n".join(module.str_report() for module in self.modules)

    def get_observables(self) -> Dict[str, Any]:
        obs = dict()
        for module in self.modules:
            obs.update(module.get_observables())
        return obs


class DefaultAccumulator(Accumulator):
    def __init__(self,
                 data: LayerData | None = None,
                 dtype: DTypeLike | None = None,
                 target_real: float | None = None,
                 target_imag: float | None = None,
                 **kwargs):
        modules = [
            IntegrationResult(data, dtype,
                              target_real=target_real, target_imag=target_imag),
            MaxWeight(data, dtype),
            FailuresMonitor(data, dtype),
            ProcessingTimes(data, dtype),
        ]
        super().__init__(modules)


class TrainingAccumulator(Accumulator):
    def __init__(self,
                 data: LayerData,
                 dtype: DTypeLike | None = None,
                 target_real: float | None = None,
                 target_imag: float | None = None,
                 training_phase: Literal['real', 'imag', 'abs'] = 'real',
                 **kwargs):
        modules = [
            IntegrationResult(data, dtype, target_real, target_imag),
            MaxWeight(data, dtype),
            FailuresMonitor(data, dtype),
            ProcessingTimes(data, dtype),
            TrainingData(data, dtype, training_phase),
        ]
        super().__init__(modules)


class IntegrationResult(AccumulatorModule):
    """
    Implements AccumulatorModule. Calculates the integration result.

    :var n_points:
    """

    def __init__(self,
                 data: LayerData | None = None,
                 dtype: DTypeLike | None = None,
                 target_real: float | None = None,
                 target_imag: float | None = None,
                 precision: int = 2,):
        self.dtype = np.dtype(np.float64) if dtype is None else dtype
        self.target_real = target_real
        self.target_imag = target_imag
        self.precision = precision
        if data is None:
            self.n_points = 0
            self.real_central_value = np.zeros(1, dtype=self.dtype)
            self.real_error = np.zeros(1, dtype=self.dtype)
            self.imag_central_value = np.zeros(1, dtype=self.dtype)
            self.imag_error = np.zeros(1, dtype=self.dtype)
        else:
            self.dtype = data.dtype
            self.n_points = data.n_points
            m = data.success
            total_wgt = data.jac[m]*data.wgt[m]*data.func_val[m]
            self.real_central_value = total_wgt.real.mean()
            self.real_error = total_wgt.real.std() / np.sqrt(self.n_points)
            self.imag_central_value = total_wgt.imag.mean()
            self.imag_error = total_wgt.imag.std() / np.sqrt(self.n_points)

    def combine_with(self: 'IntegrationResult', other: 'IntegrationResult') -> None:
        n_combined = self.n_points + other.n_points
        self.real_central_value = (self.n_points*self.real_central_value
                                   + other.n_points*other.real_central_value) / n_combined
        self.imag_central_value = (self.n_points*self.imag_central_value +
                                   other.n_points*other.imag_central_value) / n_combined
        self.real_error = np.sqrt(self.n_points**2 * self.real_error**2
                                  + other.n_points**2 * other.real_error**2) / n_combined
        self.imag_error = np.sqrt(self.n_points**2 * self.imag_error**2
                                  + other.n_points**2 * other.imag_error**2) / n_combined

        self.n_points = n_combined

    def str_report(self) -> str:
        report = [125*"="]
        report.append(
            f"Integration Result after {Colour.GREEN}{self.n_points}{Colour.END} evaluations:")
        if not self.real_error <= 0.:
            report.append(f"    {Colour.DARKCYAN}RE{Colour.END} : {
                error_fmter(self.real_central_value, self.real_error, self.precision)}")
            err_perc = abs(100. * self.real_error / self.real_central_value)
            err_col = Colour.GREEN if err_perc < 1. else Colour.RED
            rsd = err_perc / 100. * np.sqrt(self.n_points)
            report[-1] += f" ({err_col}{err_perc:.3f}%{Colour.END}, RSD={rsd:.3f})"
            if self.target_real is not None:
                diff = self.real_central_value - self.target_real
                rel_diff_perc = abs(100. * diff / self.target_real)
                report.append(
                    f"    vs target : {self.target_real:<+25.16e} Δ = {diff:<+12.2e}")
                diff_col = Colour.GREEN if rel_diff_perc < 1. else Colour.RED
                report[-1] += f" ({diff_col}{rel_diff_perc:.3f}%{Colour.END})"

        if not self.imag_error <= 0.:
            report.append(f"    {Colour.DARKCYAN}IM{Colour.END} : {
                error_fmter(self.imag_central_value, self.imag_error, self.precision)}")
            err_perc = abs(100. * self.imag_error / self.imag_central_value)
            err_col = Colour.GREEN if err_perc < 1. else Colour.RED
            rsd = err_perc / 100. * np.sqrt(self.n_points)
            report[-1] += f" ({err_col}{err_perc:.3f}%{Colour.END}, RSD={rsd:.3f})"
            if self.target_imag is not None:
                diff = self.imag_central_value - self.target_imag
                rel_diff_perc = abs(100. * diff / self.target_imag)
                report.append(
                    f"    vs target : {self.target_imag:<+25.16e} Δ = {diff:<+12.2e}")
                diff_col = Colour.GREEN if rel_diff_perc < 1. else Colour.RED
                report[-1] += f" ({diff_col}{rel_diff_perc:.3f}%{Colour.END})"

        return "\n".join(f"| > {line}" for line in report)

    def get_observables(self) -> Dict[str, Any]:
        return dict(
            real_central_value=self.real_central_value,
            imag_central_value=self.imag_central_value,
            real_error=self.real_error,
            imag_error=self.imag_error,
        )


class MaxWeight(AccumulatorModule):
    def __init__(self,
                 data: LayerData | None = None,
                 dtype: DTypeLike | None = None,):
        self.dtype = np.dtype(np.float64) if dtype is None else dtype
        if data is None:
            self.real_pos_max_wgt = np.zeros(1, dtype=self.dtype)
            self.real_neg_max_wgt = np.zeros(1, dtype=self.dtype)
            self.imag_pos_max_wgt = np.zeros(1, dtype=self.dtype)
            self.imag_neg_max_wgt = np.zeros(1, dtype=self.dtype)
            self.real_pos_max_wgt_point = None
            self.real_neg_max_wgt_point = None
            self.imag_pos_max_wgt_point = None
            self.imag_neg_max_wgt_point = None
        else:
            self.dtype = data.dtype
            m = data.success
            total_wgt = data.jac[m]*data.wgt[m]*data.func_val[m]
            momenta = data.momenta[m]
            real_pos_idx = total_wgt.real.argmax(axis=0)
            self.real_pos_max_wgt = total_wgt.real[real_pos_idx][0]
            self.real_pos_max_wgt_point = momenta[real_pos_idx][0]
            real_neg_idx = total_wgt.real.argmin(axis=0)
            self.real_neg_max_wgt = total_wgt.real[real_neg_idx][0]
            self.real_neg_max_wgt_point = momenta[real_neg_idx][0]
            imag_pos_idx = total_wgt.imag.argmax(axis=0)
            self.imag_pos_max_wgt = total_wgt.imag[imag_pos_idx][0]
            self.imag_pos_max_wgt_point = momenta[imag_pos_idx][0]
            imag_neg_idx = total_wgt.imag.argmin(axis=0)
            self.imag_neg_max_wgt = total_wgt.imag[imag_neg_idx][0]
            self.imag_neg_max_wgt_point = momenta[imag_neg_idx][0]

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
                self.imag_pos_max_wgt_point = other.real_pos_max_wgt_point
        if other.imag_neg_max_wgt_point is not None:
            if other.imag_neg_max_wgt < self.imag_neg_max_wgt:
                self.imag_neg_max_wgt = other.imag_neg_max_wgt
                self.imag_neg_max_wgt_point = other.imag_neg_max_wgt_point

    def str_report(self) -> str:
        report = ["Max weights and their location in momentum-space:"]
        if self.real_pos_max_wgt > 0.:
            report.append(
                f"    {Colour.DARKCYAN}RE[+]{Colour.END} : {self.real_pos_max_wgt[0]:<15.12e} at {self.real_pos_max_wgt_point}")
        if self.real_neg_max_wgt < 0.:
            report.append(
                f"    {Colour.DARKCYAN}RE[-]{Colour.END} : {self.real_neg_max_wgt[0]:<15.12e} at {self.real_neg_max_wgt_point}")
        if self.imag_pos_max_wgt > 0.:
            report.append(
                f"    {Colour.DARKCYAN}IM[+]{Colour.END} : {self.imag_pos_max_wgt[0]:<15.12e} at {self.imag_pos_max_wgt_point}")
        if self.imag_neg_max_wgt < 0.:
            report.append(
                f"    {Colour.DARKCYAN}IM[-]{Colour.END} : {self.imag_neg_max_wgt[0]:<15.12e} at {self.imag_neg_max_wgt_point}")
        return "\n".join(f"| > {line}" for line in report)

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
                 dtype: DTypeLike | None = None,):
        self.dtype = np.dtype(np.float64) if dtype is None else dtype
        if data is None:
            self.processing_times = dict()
            self.n_points = 0
            self._t_init = perf_counter()
        else:
            self.processing_times = data.get_processing_times()
            self.n_points = data.n_points
            self._t_init = data._t_init

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
        total_time = sum(time for time in self.processing_times.values())
        report = [
            f"Total elapsed time: {time_since_init:.2f} s | {total_time:.2f} CPU-s | {total_time*mus_factor:.1f} µs / eval. Breakdown by subroutine:"]
        subroutine_times = []
        for identifier, time in self.processing_times.items():
            perc_time = 100. * time / total_time
            if perc_time > 50.:
                subroutine_times.append(
                    f"{identifier}: {Colour.RED}{mus_factor*time:.1f}{Colour.END} µs ({Colour.RED}{int(perc_time)}%{Colour.END})")
            else:
                subroutine_times.append(
                    f"{identifier}: {mus_factor*time:.1f} µs ({int(perc_time)}%)")
        report.append(" | ".join(subroutine_times))

        return "\n".join(f"| > {line}" for line in report)

    def get_observables(self) -> Dict[str, Any]:
        return dict(
            processing_times=self.processing_times
        )


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
            return f"""| > {Colour.GREEN}No Failures, successfully evaluated all {
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

        return "\n".join(f"| > {line}" for line in report)

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
                 dtype: DTypeLike | None = None,
                 training_phase: Literal['real', 'imag', 'abs'] = 'real',):
        self.dtype = dtype
        self.training_phase = training_phase
        self.training_result: list[NDArray] = []
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
        return f"| > Trained on phase: {Colour.GREEN}{self.training_phase.upper()}{Colour.END}"

    def get_observables(self) -> Dict[str, Any]:
        return dict(
            training_phase=self.training_phase,
        )
