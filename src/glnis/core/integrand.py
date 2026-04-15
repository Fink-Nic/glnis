# type: ignore
import warnings
import math
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import wait
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Dict, List, Sequence, Callable, Literal, Any
from copy import deepcopy

from glnis.core.parameterisation import LayeredParameterisation
from glnis.core.accumulator import (
    Accumulator, TrainingAccumulator, TrainingData, GraphProperties, LayerData, IntegrationResult
)
from glnis.utils.helpers import chunks, shell_print, verify_path

try:
    import kaapos.samplers as ksamplers
    import kaapos.integrands as kintegrands
except:
    raise ImportError("Failed to import thermal integrand module.")


class Integrand(ABC):
    IDENTIFIER = "ABCIntegrand"
    momentum_space = True

    def __init__(self,
                 graph_properties: GraphProperties | List[GraphProperties],
                 continuous_dim: int = 0,
                 discrete_dims: List[int] = None,
                 use_f128: bool = False,
                 target: IntegrationResult | None = None,
                 training_phase: Literal['real', 'imag', 'abs'] = 'real',
                 **uncaught_kwargs):
        self.graph_properties = graph_properties if isinstance(graph_properties, list) else [graph_properties]
        self.continuous_dim = continuous_dim
        self.discrete_dims = discrete_dims or []
        self.use_f128 = use_f128
        self.dtype = np.dtype(
            np.float128) if self.use_f128 else np.dtype(np.float64)
        self.training_phase = training_phase
        self.target = target if target is not None else IntegrationResult()

    @abstractmethod
    def _evaluate_batch(self, continuous: NDArray, discrete: NDArray) -> NDArray:
        pass

    def evaluate_batch(self, layer_input: LayerData) -> LayerData:
        n_failed = layer_input.n_points - layer_input.success.sum()
        if n_failed > 0:
            warnings.warn(f"{n_failed} failed points detected in integrand evaluation.", RuntimeWarning)
            f = np.zeros(layer_input.n_points, dtype=np.complex128)
            f[layer_input.success.ravel()] = self._evaluate_batch(
                layer_input.momenta[layer_input.success.ravel()],
                layer_input.discrete[layer_input.success.ravel()]).ravel()
            layer_input.func_val = f
        else:
            layer_input.func_val = self._evaluate_batch(
                layer_input.momenta, layer_input.discrete)
        layer_input.update(self.IDENTIFIER)

        return layer_input

    def discrete_prior_prob_function(self, indices: NDArray, _: int = 0) -> NDArray:
        """
        Implements a default flat prior.
        """
        num_disc_input = indices.shape[1]
        if num_disc_input == len(self.discrete_dims):
            return np.zeros_like(indices, dtype=np.float64)

        disc_dim = self.discrete_dims[num_disc_input]
        return np.ones((len(indices), disc_dim), dtype=np.float64) / disc_dim

    def __call__(self, layer_input: LayerData) -> LayerData:
        return self.evaluate_batch(layer_input)

    @staticmethod
    def get_integrand_instance(graph_properties: GraphProperties | List[GraphProperties],
                               integrand_kwargs: Dict[str, Any]) -> 'Integrand':
        integrand_kwargs = deepcopy(integrand_kwargs)
        integrand_kwargs.update(dict(graph_properties=graph_properties))
        integrand_type = integrand_kwargs.pop('integrand_type')
        match integrand_type:
            case 'test':
                return TestIntegrand(**integrand_kwargs)
            case 'gammaloop':
                return GammaLoopIntegrand(**integrand_kwargs)
            case 'kaapo':
                return KaapoIntegrand(**integrand_kwargs)
            case _:
                raise NotImplementedError(
                    f"Integrand {integrand_type} has not been implemented.")


class TestIntegrand(Integrand):
    """
    Implements a normalized multivariate gaussian that integrates to unity.
    """
    IDENTIFIER = "test integrand"

    def __init__(self, offset: NDArray | List[List[float]] | List[float] | None = None,
                 sigma: float = 10.0,
                 const_f: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.offset = offset
        if offset is not None:
            self.offset = np.array(offset)
        self.sigma = sigma
        self.const_f = const_f
        if not self.const_f:
            self.target = IntegrationResult(real_central_value=1)

    def _evaluate_batch(self, continuous: NDArray, discrete: NDArray) -> NDArray:
        if self.const_f:
            return np.ones((len(continuous), 1), dtype=self.dtype)

        if self.offset is not None:
            continuous -= self.offset.reshape(1, -1)
        norm_factor = (2*np.pi * self.sigma **
                       2)**(continuous.shape[1]/2)

        return np.exp(-(continuous**2).sum(axis=1) / self.sigma**2 / 2) / norm_factor


class GammaLoopIntegrand(Integrand):
    """
    Implements a callable gammaloop state integrand.
    """
    IDENTIFIER = "gammaloop integrand"

    def __init__(self,
                 gammaloop_state_path: str,
                 process_id: int = 0,
                 integrand_name: str = "",
                 run_commands: str | List[str] = "",
                 momentum_space: bool = True,
                 sample_lmbs: bool = False,
                 sample_orientations: bool = False,
                 use_arb_prec: bool = False,
                 minimal_output: bool = True,
                 read_only_state: bool = True,
                 **kwargs):
        try:
            import gammaloop
        except:
            raise ImportError("Failed to import gammaloop module.")
        super().__init__(**kwargs)
        self.momentum_space = momentum_space
        self.use_arb_prec = use_arb_prec
        self.minimal_output = minimal_output

        self.gammaloop_state = gammaloop.GammaLoopAPI(
            gammaloop_state_path,
            level=gammaloop.LogLevel.Off,
            logfile_level=gammaloop.LogLevel.Off,
            read_only_state=read_only_state)
        if run_commands:
            if isinstance(run_commands, str):
                run_commands = [run_commands]
            for cmd in run_commands:
                self.gammaloop_state.run(cmd)
        self.outputs = dict()
        for o in self.gammaloop_state.list_outputs():
            self.outputs.update(o)
        if integrand_name not in self.outputs or integrand_name == "summed":
            integrand_name = list(self.outputs)[0]
            process_id = self.outputs[integrand_name]
        else:
            process_id = self.outputs[integrand_name]
            self.outputs = {integrand_name: process_id}
        self.sample_graphs = len(self.graph_properties) > 1
        if self.sample_graphs and self.momentum_space:
            raise ValueError("Discrete sampling over graphs is not supported in momentum space.")
        self.discrete_dims = []
        self._discrete_types: List[str] = []
        self._gl_discrete_types: Dict[str, int] = dict(graph=0)
        if self.sample_graphs:
            self.discrete_dims.append(len(self.graph_properties))
            self._discrete_types.append("graph")
        if sample_orientations:
            for itg_name, pid in self.outputs.items():
                self.gammaloop_state.run(
                    f"set process -p {pid} -i {itg_name} kv sampling.orientations='monte_carlo'")
            n_orientations = max([gp.n_orientations for gp in self.graph_properties])
            if n_orientations > 1:
                self.discrete_dims.append(n_orientations)
                self._discrete_types.append("orientation")
            self._gl_discrete_types["orientation"] = len(self._gl_discrete_types)
        else:
            for itg_name, pid in self.outputs.items():
                self.gammaloop_state.run(
                    f"set process -p {pid} -i {itg_name} kv sampling.orientations='summed'")
        if sample_lmbs:
            for itg_name, pid in self.outputs.items():
                self.gammaloop_state.run(
                    f"set process -p {pid} -i {itg_name} kv sampling.lmb_multichanneling=true")
                self.gammaloop_state.run(
                    f"set process -p {pid} -i {itg_name} kv sampling.lmb_channels='monte_carlo'")
            n_channels = max([gp.n_channels for gp in self.graph_properties])
            self._mask_lmb_dims = n_channels == 1
            if not self._mask_lmb_dims:
                self.discrete_dims.append(n_channels)
                self._discrete_types.append("lmb_channel")
            self._gl_discrete_types["lmb_channel"] = len(self._gl_discrete_types)
        else:
            for itg_name, pid in self.outputs.items():
                self.gammaloop_state.run(
                    f"set process -p {pid} -i {itg_name} kv sampling.lmb_multichanneling=false")
                self.gammaloop_state.run(
                    f"set process -p {pid} -i {itg_name} kv sampling.lmb_channels='summed'")

    def _evaluate_batch(self, continuous: NDArray, discrete: NDArray) -> NDArray:
        # When not sampling graphs, prepend graph_id=0 for gammaloop
        discrete_dims = np.zeros((len(continuous), len(self._gl_discrete_types)), dtype=np.uint64)
        for dim, tpe in enumerate(self._discrete_types):
            gl_dim = self._gl_discrete_types[tpe]
            discrete_dims[:, gl_dim] = discrete[:, dim]
        numpy_res = np.zeros(len(continuous), dtype=np.complex128)
        for itg_name, pid in self.outputs.items():
            res = self.gammaloop_state.evaluate_samples(
                points=continuous.astype(np.float64), discrete_dims=discrete_dims,
                momentum_space=self.momentum_space,
                process_id=pid,
                integrand_name=itg_name,
                use_arb_prec=self.use_arb_prec,
                minimal_output=self.minimal_output,
            )
            itg_res = np.array([s.integrand_result for s in res.samples])
            if not self.momentum_space:
                itg_res *= np.array([s.parameterization_jacobian for s in res.samples])
            numpy_res += itg_res
        return numpy_res.reshape(-1, 1)

    def discrete_prior_prob_function(self, indices: NDArray, _: int = 0) -> NDArray:
        """
        Implements a default flat prior, varying the dim depending on sampled graph
        """
        if not self.sample_graphs:
            return super().discrete_prior_prob_function(indices, _)

        num_disc_input = indices.shape[1]
        if num_disc_input == len(self.discrete_dims):
            return np.zeros_like(indices, dtype=np.float64)

        disc_type = self._discrete_types[num_disc_input]
        match disc_type:
            case "graph":
                n_graphs = self.discrete_dims[num_disc_input]
                prior = np.ones((len(indices), n_graphs), dtype=np.float64)
            case "orientation":
                max_orientations = self.discrete_dims[num_disc_input]
                n_or_arr = np.array([gp.n_orientations for gp in self.graph_properties])
                n_per_graph = n_or_arr[indices[:, 0]]
                prior = (np.arange(max_orientations) < n_per_graph[:, None]).astype(np.float64)
            case "lmb_channel":
                max_channels = self.discrete_dims[num_disc_input]
                n_ch_arr = np.array([gp.n_channels for gp in self.graph_properties])
                n_per_graph = n_ch_arr[indices[:, 0]]
                prior = (np.arange(max_channels) < n_per_graph[:, None]).astype(np.float64)
            case _:
                raise ValueError(
                    f"Invalid type of discrete input dimensions: {disc_type}. Expected one of {self._discrete_types}.")

        return prior / np.prod(prior, axis=1, keepdims=True)


class KaapoIntegrand(Integrand):
    """
    Implements a callable thermal integrand.
    """
    IDENTIFIER = "kaapo integrand"

    def __init__(self,
                 path_to_example: str,
                 params: List[float],
                 use_prec: bool = True,
                 apply_pi_factor: bool = True,
                 symbolica_integrand_kwargs: Dict[str, Any] = dict(
                     force_rebuild=False,
                     stability_tolerance=1e-3,
                     stability_abs_tolerance=1e-15,
                     stability_abs_threshold=1e-12,
                     escalate_large_weight_multiplier=-1,
                     escalate_small_momentum_multiplier=1e-2,
                     escalate_large_momentum_multiplier=1e3,
                     rotation_seed=1337,
                     n_shots=3,
                     build_eagerly=False,
                     sum_orientations=True,
                     runtime_summation=False,
                 ),
                 symbolica_integrand_prec_kwargs: Dict[str, Any] = dict(
                     prec=200,
                     stability_tolerance=1e-3,
                     stability_abs_tolerance=1e-15,
                     stability_abs_threshold=1e-12,
                     escalate_large_weight_multiplier=-1,
                     escalate_small_momentum_multiplier=-1,
                     escalate_large_momentum_multiplier=-1,
                     rotation_seed=1337,
                     n_shots=1,
                     sum_orientations=True,
                     runtime_summation=False,
                 ),
                 **kwargs):
        self.path_to_example = str(verify_path(path_to_example))
        self.params = np.array(params)
        self.symbolica_integrand_kwargs = symbolica_integrand_kwargs
        self.symbolica_integrand_prec_kwargs = symbolica_integrand_prec_kwargs
        self.use_prec = use_prec
        self.apply_pi_factor = apply_pi_factor
        self.integrand_fast = kintegrands.SymbolicaIntegrand(
            path_to_example=self.path_to_example,
            params=self.params,
            **self.symbolica_integrand_kwargs,
        )
        self.integrand_prec = kintegrands.SymbolicaIntegrandPrec(
            path_to_example=self.path_to_example,
            params=self.params,
            **self.symbolica_integrand_prec_kwargs,
        )
        self.stack = kintegrands.StableStack([
            kintegrands.PrecisionLevel(
                integrand=self.integrand_fast, level_id=0),
            kintegrands.PrecisionLevel(
                integrand=self.integrand_prec, level_id=1),
        ])
        super().__init__(**kwargs)

    def _evaluate_batch(self, continuous: NDArray, discrete: NDArray) -> NDArray:

        n_points = continuous.shape[0]
        loop_momenta = continuous.reshape(
            n_points, self.integrand_fast.n_loops, self.integrand_fast.dim).astype(np.float64)
        input = ksamplers.SamplerResult(
            weight_array=np.ones((n_points, 1)),
            jacobian_array=np.ones((n_points)),
            loop_momentum_array=loop_momenta
        )

        if not self.use_prec:
            output = self.integrand_fast.evaluate(input)
        else:
            output = self.stack.evaluate(input)

        output = np.array(output.values, dtype=self.dtype)
        if self.apply_pi_factor:
            output *= (2*np.pi)**-(3*self.integrand_fast.n_loops)

        return output


class ParameterisedIntegrand:
    def __init__(self,
                 graph_properties: GraphProperties | List[GraphProperties],
                 param_kwargs: List[Dict[str, Any]],
                 integrand_kwargs: Dict[str, Any],
                 verbose: bool = False,
                 **kwargs):
        self.graph_properties = graph_properties if isinstance(graph_properties, list) else [graph_properties]
        self.param_kwargs = param_kwargs
        self.integrand_kwargs = integrand_kwargs
        self.condition_integrand_first = integrand_kwargs.pop('condition_integrand_first', False)
        self.sum_channels = integrand_kwargs.get('sum_channels', False)
        self.verbose = verbose

        self.integrand = Integrand.get_integrand_instance(self.graph_properties, self.integrand_kwargs)
        # If the integrand is not in momentum space, we assume it does not require a parameterisation.
        if not self.integrand.momentum_space:
            self.param_kwargs = []
        self.param = LayeredParameterisation(self.graph_properties, self.param_kwargs)
        self.sum_channels = self.sum_channels and len(self.param.discrete_dims) == 1
        self.dtype = self.integrand.dtype
        self.continuous_dim = self._get_continuous_dims()
        self.discrete_dims = self._get_discrete_dims()

        self.training_phase = self.integrand.training_phase
        self.target = self.integrand.target

    def eval_integrand(self, layer_input: LayerData,
                       acc_type: Literal['default', 'training'] = 'default') -> Accumulator:
        untouched_discrete = layer_input.discrete
        if self.sum_channels:
            if len(self.param.discrete_dims) != 1:
                raise ValueError(
                    "sum_channels is only supported for a single discrete dimension in the parameterisation.")
            func_val = np.zeros((layer_input.n_points, 1), dtype=np.complex128)
            for ch in range(self.param.discrete_dims[0]):
                channels = np.full((layer_input.n_points, 1), ch, dtype=np.uint64)
                jac, mom, _ = self.param.param._layer_parameterise(layer_input.continuous, channels)
                layer_input.jac, layer_input.momenta = jac, mom
                layer_input.update(self.param.IDENTIFIER)
                integration_result = self.integrand.evaluate_batch(layer_input)
                func_val += integration_result.func_val * integration_result.jac
            layer_input.func_val = func_val
            layer_input.jac = np.ones_like(func_val, dtype=self.dtype)
            layer_input.update('processing')
            integration_result = layer_input
        else:
            pass_disc_to_integrand = np.zeros(
                (layer_input.n_points, 0), dtype=np.uint64)
            if self.condition_integrand_first:
                n_disc_integrand = len(self.integrand.discrete_dims)
                pass_disc_to_integrand = layer_input.discrete[:, :n_disc_integrand]
                layer_input.discrete = layer_input.discrete[:, n_disc_integrand:]
                layer_input.update('processing')
            parameterised = self.param.parameterise(layer_input)
            if self.condition_integrand_first:
                parameterised.discrete = np.hstack(
                    [parameterised.discrete, pass_disc_to_integrand])
                parameterised.update('processing')
            integration_result = self.integrand.evaluate_batch(parameterised)
            integration_result.discrete = untouched_discrete
            integration_result.update('processing')
        acc_kwargs = dict(
            target=self.integrand.target,
            training_phase=self.integrand.training_phase,)
        accumulator = integration_result.accumulate(acc_type, **acc_kwargs)

        if self.verbose:
            shell_print(accumulator.str_report())

        return accumulator

    def discrete_prior_prob_function(self, indices: NDArray, dim: int = 0) -> NDArray:
        if self.sum_channels:
            return self.integrand.discrete_prior_prob_function(indices, dim)

        if self.condition_integrand_first:
            n_dim = len(self.integrand.discrete_dims)
            prior1: Callable = self.integrand.discrete_prior_prob_function
            prior2: Callable = self.param.discrete_prior_prob_function
        else:
            n_dim = len(self.param.discrete_dims)
            prior1: Callable = self.param.discrete_prior_prob_function
            prior2: Callable = self.integrand.discrete_prior_prob_function

        if dim < n_dim:
            return prior1(indices, dim)

        indices = indices[:, n_dim:]
        dim -= n_dim

        return prior2(indices, dim)

    def _get_discrete_dims(self) -> List[int]:
        if self.sum_channels:
            return self.integrand.discrete_dims

        if self.condition_integrand_first:
            return self.integrand.discrete_dims + self.param.discrete_dims

        return self.param.discrete_dims + self.integrand.discrete_dims

    def _get_continuous_dims(self) -> int:
        return self.integrand.continuous_dim + self.param.continuous_dim

    def set_attribute(self, attr_name: str, value: Any) -> None:
        if hasattr(self.integrand, attr_name):
            setattr(self.integrand, attr_name, value)
        if hasattr(self.param, attr_name):
            setattr(self.param, attr_name, value)


class MPIntegrand(ParameterisedIntegrand):
    N_UNUSED = 2
    MAX_CHUNK_SIZE = 100_000
    MIN_CHUNK_SIZE = 10
    IDENTIFIER = "multiprocessing integrand"

    def __init__(self,
                 graph_properties: GraphProperties | List[GraphProperties],
                 param_kwargs: List[Dict[str, Any]],
                 integrand_kwargs: Dict[str, Any],
                 n_cores: int = 1,
                 n_shards: int = 32,
                 verbose: bool = False,
                 **kwargs):
        self.n_cores = min(n_cores, mp.cpu_count() - self.N_UNUSED) if n_cores > 0 else 1
        n_shards = min(n_shards, self.n_cores)
        self._ended = False

        ctx = mp.get_context()
        self.q_in = [ctx.Queue() for _ in range(n_shards)]
        self.q_out = [ctx.Queue() for _ in range(n_shards)]
        self.stop_event = ctx.Event()
        super().__init__(graph_properties,
                         param_kwargs,
                         integrand_kwargs,
                         verbose, **kwargs)
        worker_args = (
            self.stop_event,
            graph_properties,
            param_kwargs,
            integrand_kwargs,
        )
        self.workers = [
            ctx.Process(
                target=self._integrand_worker,
                args=((self.q_in[w_id % n_shards], self.q_out[w_id % n_shards]), *worker_args),
                daemon=True) for w_id in range(self.n_cores)
        ]
        for w in self.workers:
            w.start()
        for w_id in range(self.n_cores):
            output = self.q_out[w_id % n_shards].get()
            if output == 'STARTED':
                if self.verbose:
                    shell_print(f"Core {w_id} has been initialized.")
            else:
                raise ValueError(
                    f"Unexpected initialization value in queue: {output}")

    @staticmethod
    def _integrand_worker(queues: Sequence[mp.Queue],
                          stop_event,
                          graph_properties: GraphProperties,
                          param_kwargs: List[Dict[str, Any]],
                          integrand_kwargs: Dict[str, Any],) -> None:
        integrand = ParameterisedIntegrand(
            graph_properties,
            param_kwargs,
            integrand_kwargs,
        )
        q_in, q_out = queues
        q_out.put('STARTED')
        while not stop_event.is_set():
            try:
                data = q_in.get(timeout=0.5)
                # if data == 'STOP':
                #     break
            except:
                continue
            job_type, chunk_id, args = data
            job_type: str
            match job_type.lower():
                case 'eval':
                    layer_input, acc_type = args
                    layer_input: LayerData
                    layer_input.wake()
                    res = integrand.eval_integrand(layer_input, acc_type)
                    q_out.put((chunk_id, res))
                case 'prior':
                    indices, dim = args
                    res = integrand.discrete_prior_prob_function(indices, dim)
                    q_out.put((chunk_id, res))
                case 'set_attr':
                    attr_name, value = args
                    integrand.set_attribute(attr_name, value)
                case _:
                    shell_print("CRITICAL WARNING:")
                    shell_print(
                        f"Integrand worker has received invalid job type \"{job_type.upper()}\"")
                    shell_print("Consider terminating the program.")

    def eval_integrand(self, layer_input: LayerData,
                       acc_type: Literal['default', 'training'] = 'default') -> Accumulator:
        job_type = 'eval'

        n_cores = max(min(math.floor(layer_input.n_points /
                      self.MIN_CHUNK_SIZE), self.n_cores), 1)
        chunks_per_worker = math.ceil(
            layer_input.n_points / n_cores / self.MAX_CHUNK_SIZE)
        n_chunks = n_cores * chunks_per_worker

        data_chunks = layer_input.as_chunks(n_chunks, n_cores)

        for chunk_id, data_chunk in enumerate(data_chunks):
            self.q_in[chunk_id % len(self.q_in)].put(
                (job_type, chunk_id, (data_chunk, acc_type)), block=False)

        chunk_id_return_order = []
        readers = [q._reader for q in self.q_out]

        n_processed = 0
        while n_processed < n_chunks:
            ready = wait(readers)
            for r in ready:
                idx = readers.index(r)
                if self.stop_event.is_set():
                    break
                try:
                    output = self.q_out[idx].get()
                except Exception:
                    continue
                chunk_id, acc = output

                if n_processed == 0:
                    accumulator: Accumulator = acc
                else:
                    accumulator.combine_with(acc)
                chunk_id_return_order.append(chunk_id)
                n_processed += 1

        if n_processed != n_chunks:
            raise RuntimeError("MPIntegrand evaluation was interrupted before all chunks were processed.")

        if self.verbose:
            shell_print(accumulator.str_report())

        if acc_type == 'training':
            accumulator: TrainingAccumulator
            result_sorted = n_chunks*[None]
            training_acc: TrainingData = accumulator.training_data
            for i, sorted_id in enumerate(chunk_id_return_order):
                result_sorted[sorted_id] = training_acc.training_result[i]
            training_acc.training_result[:] = result_sorted

        accumulator.finalise()

        return accumulator

    def discrete_prior_prob_function(self, indices: NDArray, dim: int = 0) -> NDArray:
        job_type = "prior"
        n_samples = len(indices)

        n_cores = max(min(math.ceil(
            n_samples / self.MIN_CHUNK_SIZE), self.n_cores), 1)
        chunks_per_worker = math.ceil(
            n_samples / n_cores / self.MAX_CHUNK_SIZE)
        n_chunks = n_cores * chunks_per_worker
        ind_chunks = chunks(indices, n_chunks)

        for chunk_id, ind in enumerate(ind_chunks):
            args = (ind, dim)
            self.q_in[chunk_id % len(self.q_in)].put((job_type, chunk_id, args))

        result_sorted = [None]*n_chunks
        readers = [q._reader for q in self.q_out]

        n_processed = 0
        while n_processed < n_chunks:
            ready = wait(readers)
            for r in ready:
                idx = readers.index(r)
                if self.stop_event.is_set():
                    break
                try:
                    data = self.q_out[idx].get()
                except Exception:
                    continue
                chunk_id, res = data
                result_sorted[chunk_id] = res
                n_processed += 1

        if n_processed != n_chunks:
            raise RuntimeError("MPIntegrand prior computation was interrupted before all chunks were processed.")

        return np.vstack(result_sorted)

    def set_attribute(self, attr_name: str, value: Any) -> None:
        """
        Sets an attribute in the main process and all worker processes if it exists.
        """
        super().set_attribute(attr_name, value)
        job_type = "set_attr"
        for w_id in range(self.n_cores):
            self.q_in[w_id].put((job_type, None, (attr_name, value)))

    def free(self) -> None:
        """
        Terminates all worker processes and frees resources.
        """
        if self._ended:
            return

        if self.verbose:
            shell_print("Attempting to close queues.")
        self.stop_event.set()

        for w in self.workers:
            if w.is_alive():
                w.join(timeout=5)

        alive_workers = 0
        for w in self.workers:
            if w.is_alive():
                alive_workers += 1
                w.terminate()
                w.join()

        if alive_workers > 0:
            shell_print(f"Terminated {alive_workers} / {self.n_cores} alive worker(s).")

        # Explicitly close queue pipes to avoid file-descriptor leaks
        # when constructing and tearing down many MPIntegrand instances.
        for q in self.q_in + self.q_out:
            try:
                q.close()
            except Exception:
                pass
            try:
                q.join_thread()
            except Exception:
                pass

        # Explicitly close process handles to release sentinel file descriptors.
        for w in self.workers:
            try:
                w.close()
            except Exception:
                pass

        self.integrand = None
        self.param = None
        self.graph_properties = None
        self.workers = None

        self._ended = True
        if self.verbose:
            shell_print(f"{self.IDENTIFIER.upper()} has successfully terminated.")
