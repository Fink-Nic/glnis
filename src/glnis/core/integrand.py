# type: ignore
import math
import numpy as np
import multiprocessing as mp
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Dict, List, Sequence, Callable, Literal, Any
from copy import deepcopy

from glnis.core.parameterisation import LayeredParameterisation
from glnis.core.accumulator import Accumulator, TrainingData, GraphProperties, LayerData
from glnis.utils.helpers import chunks


class Integrand(ABC):
    IDENTIFIER = "ABCIntegrand"

    def __init__(self,
                 continuous_dim: int = 0,
                 discrete_dims: List[int] = [],
                 use_f128: bool = False,
                 target_real: float | None = None,
                 target_imag: float | None = None,
                 training_phase: Literal['real', 'imag', 'abs'] = 'real',):
        self.continuous_dim = continuous_dim
        self.discrete_dims = discrete_dims
        self.use_f128 = use_f128
        self.dtype = np.dtype(
            np.float128) if self.use_f128 else np.dtype(np.float64)
        self.training_phase = training_phase
        self.target_real = None if target_real == 0. else target_real
        self.target_imag = None if target_imag == 0. else target_imag

    @abstractmethod
    def _evaluate_batch(self, continuous: NDArray, discrete: NDArray) -> NDArray:
        pass

    def evaluate_batch(self, layer_input: LayerData) -> LayerData:
        layer_input.func_val = self._evaluate_batch(
            layer_input.momenta, layer_input.discrete)
        layer_input.update(self.IDENTIFIER)

        return layer_input

    def discrete_prior_prob_function(self, indices: NDArray, _: int = 0) -> NDArray:
        num_disc_input = indices.shape[1]
        if num_disc_input == len(self.discrete_dims):
            return np.zeros_like(indices, dtype=np.float64)

        disc_dim = self.discrete_dims[num_disc_input]
        norm_factor = disc_dim - indices.shape[1]
        prior = np.ones(
            (len(indices), disc_dim), dtype=np.float64)
        if num_disc_input == 0:
            return prior / norm_factor

        rows = np.repeat(np.arange(len(indices)), num_disc_input)
        prior[rows, indices.flatten()] = 0

        return prior / norm_factor

    def __call__(self, layer_input: LayerData) -> LayerData:
        return self.evaluate_batch(layer_input)

    @staticmethod
    def get_integrand_instance(integrand_kwargs: Dict[str, Any]) -> 'Integrand':
        integrand_kwargs = deepcopy(integrand_kwargs)
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
            self.target_real = 1.

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
                 integrand_name: str = "default",
                 momentum_space: bool = True,
                 **kwargs):
        try:
            from gammaloop import GammaLoopAPI
        except:
            raise ImportError(
                "CRITICAL FAILURE: Failed to import gammaloop module.")
        self.gammaloop_state = GammaLoopAPI(gammaloop_state_path)
        self.process_id = process_id
        self.integrand_name = integrand_name
        self.momentum_space = momentum_space
        super().__init__(**kwargs)

    def _evaluate_batch(self, continuous: NDArray, discrete: NDArray) -> NDArray:
        discrete_dims = np.zeros(
            (len(continuous), 1), dtype=np.uint64)
        if discrete.shape[1] > 0:
            discrete_dims = np.hstack(discrete_dims, discrete, dtype=np.uint64)
        res, _ = self.gammaloop_state.batched_inspect(
            points=continuous.astype(np.float64), momentum_space=self.momentum_space,
            process_id=self.process_id,
            integrand_name=self.integrand_name,
            use_f128=self.use_f128,  discrete_dims=discrete_dims
        )
        return res.reshape(-1, 1)


class KaapoIntegrand(Integrand):
    """
    Implements a callable thermal integrand.
    """
    IDENTIFIER = "kaapo integrand"

    def __init__(self,
                 path_to_example: str,
                 params: List[float],
                 use_prec: bool = True,
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
        try:
            import kaapos.samplers as ksamplers
            import kaapos.integrands as kintegrands
        except:
            raise ImportError("Failed to import thermal integrand module.")
        self.path_to_example = path_to_example
        self.params = np.array(params)
        self.params = np.array([2*math.pi, 0.0, 1.0])
        self.symbolica_integrand_kwargs = symbolica_integrand_kwargs
        self.symbolica_integrand_prec_kwargs = symbolica_integrand_prec_kwargs
        self.use_prec = use_prec
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
            return np.array(output.values, dtype=self.dtype)

        output = self.stack.evaluate(input)
        return np.array(output.values, dtype=self.dtype)


class ParameterisedIntegrand:
    def __init__(self,
                 graph_properties: GraphProperties,
                 param_kwargs: List[Dict[str, Any]],
                 integrand_kwargs: Dict[str, Any],
                 condition_integrand_first: bool = False,
                 verbose: bool = False,
                 **kwargs):
        self.graph_properties = graph_properties
        self.param_kwargs = param_kwargs
        self.integrand_kwargs = integrand_kwargs
        self.condition_integrand_first = condition_integrand_first
        self.verbose = verbose

        self.param = LayeredParameterisation(
            self.graph_properties, self.param_kwargs)
        self.integrand = Integrand.get_integrand_instance(
            self.integrand_kwargs)
        self.dtype = self.integrand.dtype
        self.continuous_dim = self._get_continuous_dims()
        self.discrete_dims = self._get_discrete_dims()

        self.training_phase = self.integrand.training_phase
        self.target_real = self.integrand.target_real
        self.target_imag = self.integrand.target_imag

    def eval_integrand(self, layer_input: LayerData,
                       acc_type: Literal['default', 'training'] = 'default') -> Accumulator:
        parameterised = self.param.parameterise(layer_input)
        integration_result = self.integrand.evaluate_batch(parameterised)
        acc_kwargs = dict(
            target_real=self.integrand.target_real,
            target_imag=self.integrand.target_imag,
            training_phase=self.integrand.training_phase,)
        accumulator = integration_result.accumulate(acc_type, **acc_kwargs)

        if self.verbose:
            print(accumulator.str_report())

        return accumulator

    def discrete_prior_prob_function(self, indices: NDArray, dim: int = 0) -> NDArray:
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
        if self.condition_integrand_first:
            return self.integrand.discrete_dims + self.param.discrete_dims

        return self.param.discrete_dims + self.integrand.discrete_dims

    def _get_continuous_dims(self) -> int:
        return self.integrand.continuous_dim + self.param.continuous_dim


class MPIntegrand(ParameterisedIntegrand):
    MAX_CHUNK_SIZE = 10_000
    MIN_CHUNK_SIZE = 10
    IDENTIFIER = "multiprocessing integrand"

    def __init__(self,
                 graph_properties: GraphProperties,
                 param_kwargs: List[Dict[str, Any]],
                 integrand_kwargs: Dict[str, Any],
                 condition_integrand_first: bool = False,
                 n_cores: int = 1,
                 verbose: bool = False,
                 **kwargs):
        self.n_cores = n_cores

        ctx = mp.get_context("spawn")
        self.q_in, self.q_out, self.q_discr = [ctx.Queue() for _ in range(3)]
        self.stop_event = ctx.Event()
        worker_args = (
            (self.q_in, self.q_out, self.q_discr),
            self.stop_event,
            deepcopy(graph_properties),
            deepcopy(param_kwargs),
            deepcopy(integrand_kwargs),
            condition_integrand_first,
        )
        super().__init__(graph_properties,
                         param_kwargs,
                         integrand_kwargs,
                         condition_integrand_first,
                         verbose, **kwargs)
        for _ in range(self.n_cores):
            ctx.Process(target=self._integrand_worker,
                        args=worker_args,
                        daemon=True).start()
        for core in range(self.n_cores):
            output = self.q_out.get()
            if output == "STARTED":
                if self.verbose:
                    print(f"Core {core} has been initialized.")
            else:
                raise ValueError(
                    f"Unexpected initialization value in queue: {output}")

    @staticmethod
    def _integrand_worker(queues: Sequence[mp.Queue],
                          stop_event,
                          graph_properties: GraphProperties,
                          param_kwargs: List[Dict[str, Any]],
                          integrand_kwargs: Dict[str, Any],
                          condition_integrand_first: bool,) -> None:
        integrand = ParameterisedIntegrand(graph_properties,
                                           param_kwargs,
                                           integrand_kwargs,
                                           condition_integrand_first,)
        q_in, q_out, q_discr = queues
        q_out.put("STARTED")
        while not stop_event.is_set():
            try:
                data = q_in.get(timeout=0.5)
                if data == "STOP":
                    break
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
                    q_discr.put((chunk_id, res))
                case _:
                    print("CRITICAL WARNING:")
                    print(
                        f"Integrand worker has received invalid job type \"{job_type.upper()}\"")
                    print("Consider terminating the program.")

    def eval_integrand(self, layer_input: LayerData,
                       acc_type: Literal['default', 'training'] = 'default') -> Accumulator:
        job_type = "eval"
        n_cores = min(math.ceil(layer_input.n_points /
                      self.MIN_CHUNK_SIZE), self.n_cores)

        chunks_per_worker = math.ceil(
            layer_input.n_points / n_cores / self.MAX_CHUNK_SIZE)
        n_chunks = n_cores * chunks_per_worker
        data_chunks = layer_input.as_chunks(n_chunks)

        for chunk_id, data_chunk in enumerate(data_chunks):
            self.q_in.put(
                (job_type, chunk_id, (data_chunk, acc_type)), block=False)

        chunk_id_return_order = list(range(n_chunks))
        for idx in range(n_chunks):
            if self.stop_event.is_set():
                break
            try:
                output = self.q_out.get()
            except:
                self.end()
            chunk_id, acc = output
            if idx == 0:
                accumulator: Accumulator = acc
            else:
                accumulator.combine_with(acc)
            chunk_id_return_order[idx] = chunk_id

        if self.verbose:
            print(accumulator.str_report())

        if acc_type == 'training':
            training_acc: TrainingData = accumulator.modules[-1]
            result_sorted = [training_acc.training_result[chunk_id]
                             for chunk_id in chunk_id_return_order]
            numpy_result = np.vstack(result_sorted)
            training_acc.training_result = [numpy_result]

        return accumulator

    def discrete_prior_prob_function(self, indices: NDArray, dim: int = 0) -> NDArray:
        job_type = "prior"
        n_samples = len(indices)
        n_cores = min(math.ceil(n_samples / self.MIN_CHUNK_SIZE), self.n_cores)

        chunks_per_worker = math.ceil(
            n_samples / n_cores / self.MAX_CHUNK_SIZE)
        n_chunks = n_cores * chunks_per_worker
        ind_chunks = chunks(indices, n_chunks)

        for chunk_id, ind in enumerate(ind_chunks):
            args = (ind, dim)
            self.q_in.put((job_type, chunk_id, args))

        result_sorted = [None]*n_chunks
        for _ in range(n_chunks):
            if self.stop_event.is_set():
                break
            data = self.q_discr.get()
            chunk_id, res = data
            result_sorted[chunk_id] = res

        return np.vstack(result_sorted)

    def end(self) -> None:
        if self.stop_event.is_set():
            return

        self.stop_event.set()
        try:
            for _ in range(self.n_cores):
                self.q_in.put("STOP")
        except:
            print(f"Queues have already been closed")

        print(f"{self.IDENTIFIER.upper()} has successfully terminated.")
