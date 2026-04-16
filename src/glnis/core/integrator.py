# type: ignore
import vegas
import numpy as np
import torch
import functools
from copy import deepcopy
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, List, Callable, Literal
from numpy.typing import NDArray
from torch.types import Tensor
from symbolica import NumericalIntegrator, Sample, RandomNumberGenerator

import madnis.integrator as madnis_integrator
from glnis.core.accumulator import (
    DefaultAccumulator, TrainingAccumulator, GraphProperties, LayerData
)
from glnis.core.integrand import MPIntegrand
from glnis.core.parser import SettingsParser
from glnis.utils.helpers import shell_print, error_fmter


def _block_if_ended(method) -> None:
    @functools.wraps(method)
    def wrapper(self: 'Integrator', *args, **kwargs):
        if self._ended:
            shell_print(
                f"WARNING: Integrator {self.IDENTIFIER} has already been ended and can no longer be used.")
            return None
        return method(self, *args, **kwargs)
    return wrapper


class Integrator(ABC):
    IDENTIFIER = "ABCIntegrator"

    def __init__(self,
                 integrand: MPIntegrand,
                 seed: int | None = 1337,
                 **uncaught_kwargs):
        self.integrand = integrand
        self.continuous_dim = integrand.continuous_dim
        self.discrete_dims = integrand.discrete_dims
        self.num_discrete_dims = len(integrand.discrete_dims)
        self.input_dim = self.continuous_dim + self.num_discrete_dims
        self.dtype = integrand.dtype
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self._ended = False

    @_block_if_ended
    def get_samples(self, n_points: int, *args, **kwargs) -> LayerData:
        """Returns a LayerData object containing the samples to be fed into the integrand."""
        return self._get_samples(n_points, *args, **kwargs)

    @abstractmethod
    def _get_samples(self, n_points: int, *args, **kwargs) -> LayerData:
        pass

    @_block_if_ended
    def export_state(self) -> Any:
        """Exports the state of the integrator, e.g. for checkpointing or analysis."""
        return self._export_state()

    def _export_state(self) -> 'Integrator.IntegratorState':
        return Integrator.IntegratorState(rng_state=deepcopy(self.rng.bit_generator.state))

    @_block_if_ended
    def import_state(self, state: 'Integrator.IntegratorState'):
        """
        Imports the state of the integrator from a previous export.
        Upon importing, the integrand must be set up to match the imported state.
        """
        self._import_state(state)

    def _import_state(self, state: 'Integrator.IntegratorState'):
        self.rng.bit_generator.state = deepcopy(state.rng_state)

    @_block_if_ended
    def integrate(self,
                  n_samples: int,
                  n_start: int = 1_000_000,
                  n_increase: int = 1_000_000,
                  max_batch: int = 10_000_000,
                  progress_report: bool = True) -> DefaultAccumulator:
        n_eval = 0
        n_curr_iter = n_start
        accumulator = None
        while n_eval < n_samples:
            n = min(max_batch, n_samples - n_eval, n_curr_iter)
            layer_input = self.get_samples(n)
            if accumulator is None:
                accumulator = self.integrand.eval_integrand(layer_input)
            else:
                accumulator.combine_with(self.integrand.eval_integrand(layer_input))
            n_eval += n
            if progress_report:
                if n_eval % 1e6 == 0 and n_samples % 1e6 == 0:
                    shell_print(
                        f"Evaluated {n_eval / 1e6:.0f}M / {n_samples / 1e6:.0f}M samples using {self.IDENTIFIER}...")
                    shell_print(accumulator.str_report())
                elif n_eval % 1e3 == 0 and n_samples % 1e3 == 0:
                    shell_print(
                        f"Evaluated {n_eval / 1e3:.0f}k / {n_samples / 1e3:.0f}k samples using {self.IDENTIFIER}...")
                    shell_print(accumulator.str_report())
                else:
                    shell_print(f"Evaluated {n_eval} / {n_samples} samples using {self.IDENTIFIER}...")
                    shell_print(accumulator.str_report())
            n_curr_iter += n_increase

        return accumulator

    @_block_if_ended
    def train(self, nitn: int = 10, batch_size: int = 10000, ):
        shell_print(f"Training not available for Integrator {self.IDENTIFIER}.")

    def _cont_to_discr(self, continuous: NDArray
                       ) -> Tuple[NDArray, NDArray]:
        n = len(continuous)
        if not continuous.shape[1] == self.num_discrete_dims:
            raise ValueError(
                "Shape of sampler output does not match discrete dims of stack.")

        indices = np.zeros((n, self.num_discrete_dims), dtype=np.uint64)
        prob = np.ones((n, 1))
        for i in range(self.num_discrete_dims):
            unnorm_probs = self.integrand.discrete_prior_prob_function(
                indices[:, :i], i)
            cdf = np.cumsum(unnorm_probs, axis=1)
            norm = cdf[:, [-1]]
            cdf = cdf / norm
            r = continuous[:, [i]]  # self.rng.random((n, 1))
            indices[:, i] = np.sum(cdf < r, axis=1, dtype=np.uint64)
            prob = prob * \
                np.take_along_axis(unnorm_probs, indices[:, [i]], axis=1) / norm

        return indices, (1/prob)

    @_block_if_ended
    def init_layer_data(self, n_points: int) -> LayerData:
        return LayerData(
            n_points,
            n_mom=3*self.integrand.graph_properties[0].n_loops,
            n_cont=self.continuous_dim,
            n_disc=self.num_discrete_dims,
            dtype=self.dtype,
        )

    @_block_if_ended
    def probe_prob(self, discrete: NDArray, continuous: NDArray) -> NDArray:
        """Returns the probability of sampling the given discrete and continuous points under the current sampling distribution."""
        return self._probe_prob(discrete, continuous)

    def _probe_prob(self, discrete: NDArray, continuous: NDArray) -> NDArray:
        return np.ones((len(continuous),), dtype=self.dtype)

    @staticmethod
    def from_dicts(
            graph_properties: GraphProperties | List[GraphProperties],
            parameterisation_kwargs: List[Dict[str, Any]],
            integrand_kwargs: Dict[str, Any],
            integrator_kwargs: Dict[str, Any],) -> 'Integrator':

        n_cores = integrand_kwargs.pop('n_cores', 1)
        n_shards = integrand_kwargs.pop('n_shards', 32)
        integrator_type: str | None = integrator_kwargs.pop('integrator_type', None)

        integrand = MPIntegrand(
            graph_properties=graph_properties,
            param_kwargs=parameterisation_kwargs,
            integrand_kwargs=integrand_kwargs,
            n_cores=n_cores,
            n_shards=n_shards,
        )

        match integrator_type.lower() if integrator_type is not None else None:
            case 'naive':
                return NaiveIntegrator(integrand, **integrator_kwargs)
            case 'vegas':
                return VegasIntegrator(integrand, **integrator_kwargs)
            case 'havana':
                return HavanaIntegrator(integrand, **integrator_kwargs)
            case 'madnis':
                return MadnisIntegrator(integrand, **integrator_kwargs)
            case _:
                return NaiveIntegrator(integrand, **integrator_kwargs)

    @staticmethod
    def from_settings(settings: str | Dict) -> 'Integrator':
        Parser = SettingsParser(settings)
        graph_properties = Parser.get_graph_properties()
        parameterisation_kwargs = Parser.get_parameterisation_kwargs()
        integrand_kwargs = Parser.get_integrand_kwargs()
        integrator_kwargs = Parser.get_integrator_kwargs()

        return Integrator.from_dicts(
            graph_properties=graph_properties,
            parameterisation_kwargs=parameterisation_kwargs,
            integrand_kwargs=integrand_kwargs,
            integrator_kwargs=integrator_kwargs
        )

    @_block_if_ended
    def get_info(self) -> Dict[str, Any]:
        return self._get_info()

    def _get_info(self) -> Dict[str, Any]:
        info = {
            "Integrator type": self.IDENTIFIER,
            "Training phase": self.integrand.training_phase,
            "Input dimension": self.input_dim,
            "Continuous dimension": self.continuous_dim,
            "Discrete dimension": self.discrete_dims,
            "Random seed": self.seed
        }
        return info

    @_block_if_ended
    def display_info(self) -> None:
        info = self.get_info()
        shell_print("Integrator information:")
        for key, value in info.items():
            shell_print(f"    {key}: {value}")

    def free(self) -> None:
        """Performs any necessary cleanup after integration is done."""
        if self._ended:
            return
        self.integrand.free()
        self.integrand = None
        self._ended = True

    @dataclass(kw_only=True)
    class IntegratorState:
        rng_state: Dict


class NaiveIntegrator(Integrator):
    IDENTIFIER = 'naive sampler'

    def __init__(self,
                 integrand: MPIntegrand,
                 **kwargs,):
        super().__init__(integrand=integrand, **kwargs)

    def _get_samples(self, n_points: int) -> LayerData:
        layer_input = self.init_layer_data(n_points)
        layer_input.continuous = self.rng.random((n_points, self.continuous_dim))
        discrete = self.rng.uniform(
            size=(n_points, len(self.discrete_dims)))
        layer_input.discrete, layer_input.wgt = self._cont_to_discr(discrete)
        layer_input.update(self.IDENTIFIER)

        return layer_input


class VegasIntegrator(Integrator):
    IDENTIFIER = 'vegas sampler'

    def __init__(self,
                 integrand: MPIntegrand,
                 vegas_init_kwargs: Dict[str, Any] = dict(),
                 bins: int = 100,
                 alpha: float = 0.7,
                 **kwargs):
        super().__init__(integrand=integrand, **kwargs)
        self.bins = bins
        self.vegas_init_kwargs = vegas_init_kwargs
        self.adaptive_map = vegas.AdaptiveMap(grid=self.input_dim*[[0, 1],], ninc=bins)
        self.alpha = alpha

    @_block_if_ended
    def train(self, nitn: int = 10, batch_size: int = 1000, alpha: float | None = None) -> vegas._vegas.RAvg:
        if alpha is None:
            alpha = self.alpha
        shell_print(f"Training Vegas for {nitn} iterations à {batch_size} samples with alpha={alpha}:")
        report = []
        for itn in range(nitn):
            layer_input, r = self.get_samples(batch_size, return_r=True)
            accumulated_result: TrainingAccumulator = self.integrand.eval_integrand(
                layer_input, 'training')
            f = accumulated_result.training_data.training_result[0].astype(np.float64)
            self.adaptive_map.add_training_data(r, f.ravel())
            self.adaptive_map.adapt(alpha=alpha)
            res = accumulated_result.training_data.central_value
            err = accumulated_result.training_data.error
            rsd = accumulated_result.training_data.rsd
            report.append(f"itn {itn+1}: {error_fmter(res, err)}, RSD={rsd:.3f}")

        return "\n".join(f"{line}" for line in report)

    def _get_samples(self, n_points: int, return_r: bool = False) -> LayerData:
        # Just need this for the timestamp, since vegas may decide on a different sample size.
        layer_input = self.init_layer_data(n_points)
        # adaptive_map.map will fill the containers x_all and jac with the mapped samples and weights
        r = self.rng.random(size=(n_points, self.input_dim), dtype=np.float64)
        x_all = np.empty(shape=(n_points, self.input_dim), dtype=np.float64)
        jac = np.empty(shape=(n_points,), dtype=np.float64)
        self.adaptive_map.map(r, x_all, jac)
        layer_input.continuous = x_all[:, :self.continuous_dim]
        layer_input.discrete, disc_wgt = self._cont_to_discr(x_all[:, self.continuous_dim:])
        layer_input.wgt = disc_wgt * jac.reshape(-1, 1)
        layer_input.update(self.IDENTIFIER)

        if return_r:
            return layer_input, r
        return layer_input

    def _probe_prob(self, discrete: NDArray, continuous: NDArray) -> NDArray:
        x_all = np.hstack([continuous, discrete])
        jac_no_disc = np.prod(self.adaptive_map.jac1d(x_all)[:, :self.continuous_dim], axis=1)
        jac_disc = np.ones((continuous.shape[0],), dtype=jac_no_disc.dtype)
        # Basically invert _cont_to_disc
        for i in range(self.num_discrete_dims):
            disc_dim = self.discrete_dims[i]
            unnorm_probs = self.integrand.discrete_prior_prob_function(
                discrete[:, :i], i)
            cdf = np.cumsum(unnorm_probs, axis=1)
            norm = cdf[:, -1]
            cdf = cdf / norm[:, None]
            # Need to find all intersections of the discrete cdf with the vegas linear interpolation
            g_disc = np.array(self.adaptive_map.grid[self.continuous_dim - 1 + i])
            disc_bins = np.zeros((continuous.shape[0], disc_dim+1), dtype=unnorm_probs.dtype)
            for j in range(disc_dim):
                y_intersection = cdf[:, j]
                x_intersection = np.interp(y_intersection, np.linspace(0, 1, len(g_disc)), g_disc)
                disc_bins[:, j+1] = x_intersection
            disc_jacs = np.diff(disc_bins, axis=1)
            jac_disc *= np.take_along_axis(disc_jacs, discrete[:, [i]]).ravel()

        return 1.0 / jac_no_disc / jac_disc

    def _export_state(self) -> 'VegasIntegrator.VegasState':
        return VegasIntegrator.VegasState(grid=deepcopy(self.adaptive_map.extract_grid()),
                                          rng_state=deepcopy(self.rng.bit_generator.state))

    def _import_state(self, state: 'VegasIntegrator.VegasState'):
        if not isinstance(state, VegasIntegrator.VegasState):
            raise ValueError("State for VegasIntegrator must be of type VegasState.")
        super()._import_state(state)
        self.adaptive_map = vegas.AdaptiveMap(grid=deepcopy(state.grid))
        if self.adaptive_map.dim != self.input_dim:
            raise ValueError("Imported Vegas state does not match input dimension of integrand.")

    @dataclass(kw_only=True)
    class VegasState(Integrator.IntegratorState):
        grid: List[List[float]]

    def _get_info(self) -> Dict[str, Any]:
        info = super()._get_info()
        info["bins"] = self.bins
        info["alpha"] = self.alpha
        return info


class HavanaIntegrator(Integrator):
    IDENTIFIER = "havana sampler"

    def __init__(self, integrand: MPIntegrand,
                 stream_id: int = 0,
                 use_uniform: bool = False,
                 max_prob_ratio: float = 100,
                 n_continuous_bins: int = 128,
                 discrete_learning_rate: float = 1.5,
                 continuous_learning_rate: float = 1.5,
                 **kwargs,):
        super().__init__(integrand, **kwargs)
        if self.num_discrete_dims == 0:
            self.havana = NumericalIntegrator.continuous(
                self.continuous_dim, n_continuous_bins)
            self.uniform_disc_grid = None
        elif self.num_discrete_dims == 1 and not use_uniform:
            self.havana = NumericalIntegrator.discrete(
                [NumericalIntegrator.continuous(
                    self.continuous_dim, n_continuous_bins) for _ in range(self.discrete_dims[0])],
                max_prob_ratio,
            )
            self.uniform_disc_grid = False
        else:
            self.havana = NumericalIntegrator.uniform(
                self.discrete_dims,
                NumericalIntegrator.continuous(
                    self.continuous_dim, n_continuous_bins)
            )
            self.uniform_disc_grid = True

        self.bins = n_continuous_bins
        self.max_prob_ratio = max_prob_ratio
        self.stream_id = stream_id
        self.symbolica_rng = RandomNumberGenerator(self.seed, self.stream_id)
        self.discrete_learning_rate = discrete_learning_rate
        self.continuous_learning_rate = continuous_learning_rate

    @_block_if_ended
    def train(self, nitn: int = 10, batch_size: int = 10000) -> str:
        report = [
            f"Training Havana for {nitn} iterations à {batch_size} samples:"]
        n_digits = len(str(nitn))
        for itn in range(nitn):
            samples, layer_input = self.get_samples(batch_size, True, True)
            acc: TrainingAccumulator = self.integrand.eval_integrand(
                layer_input, 'training')
            weighted_func_val = acc.training_data.training_result[0]
            self.havana.add_training_samples(
                samples, weighted_func_val.ravel().tolist())
            avg, err, chi_sq = self.havana.update(
                self.discrete_learning_rate, self.continuous_learning_rate)

            report.append(
                f"It {itn+1:0>{n_digits}}: {avg:.6e} +- {err:.6e}, chi={chi_sq:.3f}")

        return "\n".join(f"{line}" for line in report)

    def _get_samples(self, n_points: int, return_samples: bool = False, training: bool = False,
                     ) -> Tuple[List[Sample], LayerData] | LayerData:
        layer_input = self.init_layer_data(n_points)
        samples = self.havana.sample(n_points, self.symbolica_rng)

        continuous = np.zeros(
            (n_points, self.continuous_dim), dtype=self.integrand.dtype)
        for i in range(n_points):
            continuous[i] = samples[i].c

        if self.num_discrete_dims > 0:
            discrete = np.zeros(
                (n_points, self.num_discrete_dims), dtype=np.uint64)
            for i in range(n_points):
                discrete[i] = samples[i].d
        else:
            discrete = np.zeros((n_points, 0), dtype=np.uint64)

        layer_input.continuous = continuous
        layer_input.discrete = discrete
        total_wgt = np.ones((n_points), dtype=self.integrand.dtype)
        for i in range(self.num_discrete_dims):
            try:
                discrete_wgt = np.take_along_axis(
                    self.integrand.discrete_prior_prob_function(
                        discrete[:, :i], i),
                    discrete[:, [i]]).ravel()
                total_wgt /= discrete_wgt
            except Exception as e:
                shell_print(f"Error computing discrete weight for dimension {i}: {e}")
                shell_print(f"Discrete samples for this dimension: {discrete[:, i]}")
                raise e

        if training:
            total_wgt /= np.array(self.discrete_dims).prod()
        else:
            sample_weights = np.array([s.weights for s in samples])
            sample_weights = sample_weights[:, -1]
            total_wgt *= sample_weights

            # sample_weights = np.prod(sample_weights, axis=1)
            # total_wgt *= sample_weights * np.array(self.discrete_dims).prod()

        layer_input.wgt = total_wgt
        layer_input.update(self.IDENTIFIER)

        if return_samples:
            return samples, layer_input

        return layer_input

    def _probe_prob(self, discrete: NDArray, continuous: NDArray) -> NDArray:
        from symbolica import Probe
        prob = np.empty((len(continuous),), dtype=np.float64)
        if self.num_discrete_dims > 0:
            for i in range(len(continuous)):
                probe_sample = Probe.discrete(discrete[i], continuous[i].tolist())
                prob[i] = 1 / self.havana.probe(probe_sample)
        else:
            for i in range(len(continuous)):
                probe_sample = Probe.continuous(continuous[i].tolist())
                prob[i] = 1 / self.havana.probe(probe_sample)

        return prob

    def _export_state(self) -> 'HavanaIntegrator.HavanaState':
        return HavanaIntegrator.HavanaState(
            rng_state=deepcopy(self.rng.bit_generator.state),
            grid=self.havana.export_grid(export_samples=False),
            symbolica_rng_state=self.symbolica_rng.save(),
        )

    def _import_state(self, state: 'HavanaIntegrator.HavanaState'):
        if not isinstance(state, HavanaIntegrator.HavanaState):
            raise ValueError("State for HavanaIntegrator must be of type HavanaState.")
        super()._import_state(state)
        self.havana = NumericalIntegrator.import_grid(state.grid)
        self.symbolica_rng = RandomNumberGenerator.load(state.symbolica_rng_state)

    @dataclass(kw_only=True)
    class HavanaState(Integrator.IntegratorState):
        grid: bytes
        symbolica_rng_state: bytes

    def _get_info(self) -> Dict[str, Any]:
        info = super()._get_info()
        info["Stream ID"] = self.stream_id
        info["bins"] = self.bins
        info["max_prob_ratio"] = self.max_prob_ratio
        info["uniform_disc_grid"] = self.uniform_disc_grid
        info["discrete_learning_rate"] = self.discrete_learning_rate
        info["continuous_learning_rate"] = self.continuous_learning_rate
        return info


class MadnisIntegrator(Integrator):
    IDENTIFIER = 'madnis sampler'

    def __init__(
            self,
        integrand: MPIntegrand,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        use_scheduler: bool = True,
        scheduler_type: Literal["cosineannealing"] | List[str] | None = None,
        # warmup_steps: List[int] = [],
        scheduler_kwargs: Dict[str, Any] = dict(),
        loss_type: Literal["test", "variance", "variance_softclip",
                           "kl_divergence", "kl_divergence_softclip"] = "kl_divergence",
        loss_kwargs: Dict[str, Any] = dict(),
        discrete_dims_position: Literal["first", "last"] = "first",
        discrete_model: Literal["transformer",
                                "made"] = "transformer",
        transformer: Dict[str, Any] = dict(
            embedding_dim=64,
            feedforward_dim=64,
            heads=4,
            mlp_units=64,
            transformer_layers=1,),
        made: Dict[str, Any] = dict(),
        flow_kwargs: Dict[str, Any] = dict(
            uniform_latent=True,
            permutations="log",
            layers=3,
            units=32,
            bins=10,
            min_bin_width=1e-3,
            min_bin_height=1e-3,
            min_bin_derivative=1e-3,),
        pretrain_c_flow: bool = False,
        pretraining_kwargs: Dict[str, Any] = dict(
            nitn=10,
            neval=10000,
            bins_mult=4,
            alpha=0.7,
        ),
        callback: Callable[[object], None] | None = None,
        max_batch_size: int = 100_000,
        use_gpu: bool = True,
        **kwargs,
    ):
        super().__init__(integrand, **kwargs)
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(self.seed)

        self.device = torch.device('cpu')  # default

        if torch.cuda.is_available() and use_gpu:
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                if (7, 0) <= (major, minor) < (12, 0):
                    self.device = torch.device(f'cuda:{i}')
                    shell_print(
                        f"Using CUDA device {i}: {torch.cuda.get_device_name(i)} (capability {major}.{minor})")
                    break
            else:
                shell_print("CUDA devices found but none are compatible. Using CPU.")
        else:
            shell_print("No CUDA device found. Using CPU.")

        self.use_scheduler = use_scheduler
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs
        self.batch_size = batch_size
        self.discrete_dims_position = discrete_dims_position

        from glnis.core import losses
        match loss_type.lower():
            case "variance":
                loss = madnis_integrator.losses.stratified_variance
            case "variance_softclip":
                loss = losses.stratified_variance_softclip
            case "kl_divergence":
                loss = madnis_integrator.losses.kl_divergence
            case "kl_divergence_softclip":
                loss = losses.kl_divergence_softclip
            case "test":
                loss = losses.test()
            case _:
                loss = None

        def loss_with_kwargs(*args, **kwargs):
            return loss(*args, **kwargs, **loss_kwargs)

        madnis_integrand = madnis_integrator.Integrand(
            function=self._madnis_eval,
            input_dim=self.input_dim,
            discrete_dims=self.discrete_dims,
            discrete_dims_position=self.discrete_dims_position,
            discrete_prior_prob_function=self._madnis_discrete_prior_prob_function,
        )
        discrete_flow_kwargs = transformer if discrete_model == "transformer" else made

        self.madnis = madnis_integrator.Integrator(
            madnis_integrand,
            device=self.device,
            discrete_flow_kwargs=discrete_flow_kwargs,
            loss=loss_with_kwargs,
            batch_size=self.batch_size,
            discrete_model=discrete_model,
            learning_rate=learning_rate,
            flow_kwargs=flow_kwargs,
        )
        # self.madnis.optimizer = torch.optim.Adam(self.madnis.flow.parameters(), lr=learning_rate,
        #                                          weight_decay=1e-5, betas=(0.8, 0.99))
        self.callback = self._default_callback if callback is None else callback
        self.max_batch_size = max_batch_size

        if pretrain_c_flow:
            bins = int(pretraining_kwargs.get('bins_mult', 4)*flow_kwargs.get('bins', 100))
            vegas_integrator = VegasIntegrator(
                self.integrand,
                bins=bins,
                alpha=pretraining_kwargs.get('alpha', 0.7)
            )
            shell_print(f"Pretraining continuous flow...")
            nitn = pretraining_kwargs.get('nitn', 10)
            report = vegas_integrator.train(
                nitn=nitn, batch_size=pretraining_kwargs.get('neval', 10000))
            shell_print(report)
            grid = torch.as_tensor(
                vegas_integrator.adaptive_map.extract_grid(),
                device=self.madnis.dummy.device,
                dtype=self.madnis.dummy.dtype,
            )
            if self.num_discrete_dims > 0:
                if self.discrete_dims_position == "first":
                    grid = grid[self.num_discrete_dims:, :]
                elif self.discrete_dims_position == "last":
                    grid = grid[:-self.num_discrete_dims, :]
                self.madnis.flow.continuous_flow.init_with_grid(grid)
            else:
                self.madnis.flow.init_with_grid(grid)
            shell_print(f"MadNIS pretraining successfully completed!")

    @_block_if_ended
    def train(self, nitn: int = 10, _=None):
        if self.use_scheduler:
            self.madnis.scheduler = self._get_scheduler(
                self.madnis.step + nitn, self.scheduler_type)
        self.madnis.train(nitn, self.callback, True)

    def _get_samples(self, n_points: int) -> LayerData:
        layer_input = self.init_layer_data(n_points)
        # LayerData objects return a view of the fields, so we can fill them directly,
        # but this would sidestep data validation, so we fill a copy and then assign it
        continuous = np.empty((n_points, self.continuous_dim), dtype=self.integrand.dtype)
        discrete = np.empty((n_points, self.num_discrete_dims), dtype=np.uint64)
        wgt = np.empty((n_points, 1), dtype=self.integrand.dtype)

        n_eval = 0
        while n_eval < n_points:
            n = min(self.max_batch_size, n_points - n_eval)
            with torch.no_grad():
                x_all, prob = self.madnis.flow.sample(
                    n,
                    return_prob=True,
                    device=self.madnis.dummy.device,
                    dtype=self.madnis.dummy.dtype,
                )
            discrete[n_eval:n_eval+n, :], continuous[n_eval:n_eval+n, :] = self._madnis_output_to_disc_cont(x_all)
            wgt[n_eval:n_eval+n, :] = 1 / prob.numpy(force=True).reshape(-1, 1)
            n_eval += n
        layer_input.continuous = continuous
        layer_input.discrete = discrete
        layer_input.wgt = wgt
        layer_input.update(self.IDENTIFIER)

        return layer_input

    def _madnis_eval(self, x_all: Tensor) -> Tensor:
        layer_input = self.init_layer_data(x_all.shape[0])
        layer_input.discrete, layer_input.continuous = self._madnis_output_to_disc_cont(x_all)
        layer_input.update(self.IDENTIFIER)

        accumulated_result: TrainingAccumulator = self.integrand.eval_integrand(
            layer_input, 'training')
        weighted_func_val = accumulated_result.training_data.training_result[0].flatten()

        start = 2.0
        max_step = 1500
        highlight_peaks = False
        if highlight_peaks:
            exponent = start + self.madnis.step * (1.0 - start) / max_step
            exponent = max(exponent, 1.0)
            np.power(np.abs(weighted_func_val), exponent, out=weighted_func_val)

        torch_output = torch.from_numpy(
            weighted_func_val.astype(np.float64)).to(self.device)
        return torch_output

    def _probe_prob(self, discrete: NDArray, continuous: NDArray) -> NDArray:
        n_samples = len(continuous)
        if self.madnis.integrand.discrete_dims_position == "first":
            x_all = torch.as_tensor(
                np.hstack([discrete, continuous]),
                device=self.madnis.dummy.device,
                dtype=self.madnis.dummy.dtype)
        elif self.madnis.integrand.discrete_dims_position == "last":
            x_all = torch.as_tensor(
                np.hstack([continuous, discrete]),
                device=self.madnis.dummy.device,
                dtype=self.madnis.dummy.dtype)
        prob = np.empty((n_samples,), dtype=np.float64)

        n_eval = 0
        while n_eval < n_samples:
            n = min(self.max_batch_size, n_samples - n_eval)
            with torch.no_grad():
                prob[n_eval:n_eval+n] = self.madnis.flow.prob(
                    x_all[n_eval:n_eval+n, :]).numpy(force=True).reshape(-1)
            n_eval += n
        return prob

    def _madnis_discrete_prior_prob_function(self, indices: Tensor, dim: int = 0) -> Tensor:
        numpy_output = self.integrand.discrete_prior_prob_function(
            indices.numpy(force=True).astype(np.uint64), dim)
        torch_output = torch.from_numpy(
            numpy_output.astype(np.float64)).to(self.device)
        return torch_output

    def _madnis_output_to_disc_cont(self, x_all: Tensor) -> Tuple[NDArray, NDArray]:
        if self.discrete_dims_position == "first":
            discrete = x_all[:, :self.num_discrete_dims].numpy(force=True)
            continuous = x_all[:, self.num_discrete_dims:].numpy(force=True)
        else:
            discrete = x_all[:, -self.num_discrete_dims:].numpy(force=True)
            continuous = x_all[:, :-self.num_discrete_dims].numpy(force=True)
        return discrete, continuous

    def _get_scheduler(self, T_max: int, scheduler_type: str | None
                       ) -> torch.optim.lr_scheduler._LRScheduler | None:
        if scheduler_type is None:
            return None
        match scheduler_type.lower():
            case 'cosineannealing':
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.madnis.optimizer, T_max=T_max, **self.scheduler_kwargs)
            case 'reducelronplateau':
                return torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.madnis.optimizer, T_max=T_max, **self.scheduler_kwargs)
            case 'linear':
                return torch.optim.lr_scheduler.LinearLR(
                    self.madnis.optimizer, T_max=T_max, **self.scheduler_kwargs)
            case _:
                return None
        # if not isinstance(scheduler_type, list):
        #     scheduler_type = [scheduler_type]

        # sched_list = []
        # for sched in scheduler_type:
        #     match sched.lower():
        #         case 'cosineannealing':
        #             sched_list.append(torch.optim.lr_scheduler.CosineAnnealingLR)
        #         case 'reducelronplateau':
        #             sched_list.append(torch.optim.lr_scheduler.ReduceLROnPlateau)
        #         case 'linear':
        #             sched_list.append(torch.optim.lr_scheduler.LinearLR)
        #         case _:
        #             return None

        # torch.optim.lr_scheduler.LambdaLR(self.madnis.optimizer, self.)

    @staticmethod
    def _default_callback(status: madnis_integrator.TrainingStatus) -> None:
        if (status.step + 1) % 10 == 0:
            shell_print(f"Step {status.step+1}: Loss={status.loss} ")

    def _export_state(self) -> 'MadnisIntegrator.MadnisState':
        flow_state = self.madnis.flow.state_dict()
        cwnet_state = self.madnis.cwnet.state_dict() if self.madnis.cwnet is not None else None
        optimizer_state = self.madnis.optimizer.state_dict() if self.madnis.optimizer is not None else None
        scheduler_state = self.madnis.scheduler.state_dict() if self.madnis.scheduler is not None else None
        return MadnisIntegrator.MadnisState(
            rng_state=deepcopy(self.rng.bit_generator.state),
            torch_rng_state=torch.get_rng_state(),
            flow_state=flow_state,
            cwnet_state=cwnet_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
        )

    def _import_state(self, state: 'MadnisIntegrator.MadnisState'):
        if not isinstance(state, MadnisIntegrator.MadnisState):
            raise ValueError("Invalid state type for MadnisIntegrator.")
        super()._import_state(state)
        torch.set_rng_state(state.torch_rng_state)
        self.madnis.flow.load_state_dict(state.flow_state)
        if state.cwnet_state is not None:
            if self.madnis.cwnet is None:
                shell_print("WARNING: Cannot load CWNet state: Madnis integrator was not initialized with a CWNet.")
            else:
                self.madnis.cwnet.load_state_dict(state.cwnet_state)
        if state.optimizer_state is not None:
            if self.madnis.optimizer is None:
                shell_print("WARNING: Cannot load optimizer state: Madnis integrator was not initialized with an optimizer.")
            else:
                self.madnis.optimizer.load_state_dict(state.optimizer_state)
        if state.scheduler_state is not None:
            self.madnis.scheduler = self._get_scheduler(T_max=1, scheduler_type=self.scheduler_type)
            self.madnis.scheduler.load_state_dict(state.scheduler_state)

    def free(self) -> None:
        """Performs any necessary cleanup after integration is done."""
        super().free()
        # Move modules off GPU and drop references so repeated runs do not accumulate memory.
        if hasattr(self, "madnis") and self.madnis is not None:
            if self.device.type == 'cuda':
                try:
                    self.madnis.flow.to('cpu')
                    if self.madnis.cwnet is not None:
                        self.madnis.cwnet.to('cpu')
                except Exception:
                    # Best-effort cleanup; continue with reference release.
                    pass

            self.madnis.optimizer = None
            self.madnis.scheduler = None
            self.madnis = None

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()

    @dataclass(kw_only=True)
    class MadnisState(Integrator.IntegratorState):
        torch_rng_state: Tensor
        flow_state: Dict[str, Any]
        cwnet_state: Dict[str, Any] | None
        optimizer_state: Dict[str, Any] | None
        scheduler_state: Dict[str, Any] | None

    def _get_info(self) -> Dict[str, Any]:
        info = super()._get_info()
        info["scheduler_type"] = self.scheduler_type
        info["device"] = str(self.device)
        if self.num_discrete_dims > 0:
            trainable_disc_flow = sum(p.numel() for p in self.madnis.flow.discrete_flow.parameters() if p.requires_grad)
            total_disc_flow = sum(p.numel() for p in self.madnis.flow.discrete_flow.parameters())
            info["discrete flow trainable parameters"] = trainable_disc_flow
            info["discrete flow total parameters"] = total_disc_flow

            trainable_cont_flow = sum(p.numel()
                                      for p in self.madnis.flow.continuous_flow.parameters() if p.requires_grad)
            total_cont_flow = sum(p.numel() for p in self.madnis.flow.continuous_flow.parameters())
            info["continuous flow trainable parameters"] = trainable_cont_flow
            info["continuous flow total parameters"] = total_cont_flow

        trainable_flow = sum(p.numel() for p in self.madnis.flow.parameters() if p.requires_grad)
        total_flow = sum(p.numel() for p in self.madnis.flow.parameters())
        info["flow trainable parameters"] = trainable_flow
        info["flow total parameters"] = total_flow

        if self.madnis.cwnet is not None:
            trainable_cwnet = sum(p.numel() for p in self.madnis.cwnet.parameters() if p.requires_grad)
            total_cwnet = sum(p.numel() for p in self.madnis.cwnet.parameters())
            info["CWNet trainable parameters"] = trainable_cwnet
            info["CWNet total parameters"] = total_cwnet

        return info
