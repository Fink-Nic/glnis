# type: ignore
import vegas
import numpy as np
import torch
import functools
import io

from copy import deepcopy
from dataclasses import dataclass
from abc import ABC
from typing import Dict, Tuple, Any, List, Callable, Literal
from numpy.typing import NDArray
from torch.types import Tensor
from symbolica import NumericalIntegrator, Sample, RandomNumberGenerator

import madnis.integrator as madnis_integrator
from glnis.core.accumulator import DefaultAccumulator, TrainingAccumulator
from glnis.core.integrand import MPIntegrand
from glnis.core.parser import SettingsParser
from glnis.utils.helpers import shell_print, error_fmter
from glnis.utils.types import GraphProperties, LayerData


def _block_if_ended(method):
    @functools.wraps(method)
    def wrapper(self: 'Integrator', *args, **kwargs):
        if self._ended:
            import warnings
            warnings.warn(f"Integrator {self.IDENTIFIER} has already been ended and can no longer be used.")
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
        self._discrete_prod: int = int(np.prod(self.discrete_dims))
        self.input_dim = self.continuous_dim + self.num_discrete_dims
        self.dtype = integrand.dtype
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.total_training_samples = 0
        self.step = 0
        self._ended = False

    @_block_if_ended
    def get_samples(self, n_points: int, *args, **kwargs) -> LayerData:
        """Returns a LayerData object containing the samples to be fed into the integrand."""
        return self._get_samples(n_points, *args, **kwargs)

    @_block_if_ended
    def integrate(
        self,
        n_samples: int,
        n_start: int = 1_000_000,
        n_increase: int = 1_000_000,
        max_batch: int = 10_000_000,
        progress_report: bool = True,
        acc_type: Literal['default', 'training'] = 'default'
    ) -> DefaultAccumulator | TrainingAccumulator:
        n_eval = 0
        n_curr_iter = n_start
        accumulator = None
        while n_eval < n_samples:
            n = min(max_batch, n_samples - n_eval, n_curr_iter)
            layer_input = self.get_samples(n)
            if accumulator is None:
                accumulator = self.integrand.eval_integrand(layer_input, acc_type)
            else:
                accumulator.combine_with(self.integrand.eval_integrand(layer_input, acc_type))
            accumulator.finalise()
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
    def train(self, nitn: int = 10, batch_size: int = 10000,
              callback: Callable[['Integrator.TrainingStatus'], None] | None = None) -> str | None:
        shell_print(f"Training not available for Integrator {self.IDENTIFIER}.")

    @_block_if_ended
    def export_state(self) -> 'Integrator.IntegratorState':
        """Exports the state of the integrator, e.g. for checkpointing or analysis."""
        state = self._export_state()
        state.obj_dict.pop('integrand', None)
        state.obj_dict.pop('rng', None)
        return state

    @_block_if_ended
    def import_state(self, state: 'Integrator.IntegratorState'):
        """
        Imports the state of the integrator from a previous export.
        Upon importing, the integrand must be set up to match the imported state.
        """
        self.__dict__.update(state.obj_dict)
        self.rng = np.random.default_rng()  # reinitialize to reset state
        self.rng.bit_generator.state = deepcopy(state.rng_state)
        self._import_state(state)

    @_block_if_ended
    def probe_prob(self, discrete: NDArray, continuous: NDArray | None = None) -> NDArray:
        """Returns the probability of sampling the given discrete and continuous points under the current state of the sampler."""
        return self._probe_prob(discrete, continuous)

    def init_layer_data(self, n_points: int) -> LayerData:
        """
        Initializes a LayerData object with the appropriate dimensions and dtype for the integrand.
        """
        return LayerData(
            n_points,
            n_mom=3*self.integrand.graph_properties[0].n_loops,
            n_cont=self.continuous_dim,
            n_disc=self.num_discrete_dims,
            dtype=self.dtype,
        )

    @_block_if_ended
    def display_info(self) -> None:
        from glnis.utils.helpers import Colour
        info = self.get_info()
        shell_print("Integrator information:")
        for key, value in info.items():
            shell_print(f"    {Colour.CYAN}{key}{Colour.END}: {value}")

    @_block_if_ended
    def get_info(self) -> Dict[str, Any]:
        return self._get_info()

    def free(self) -> None:
        """Performs any necessary cleanup after integration is done."""
        if self._ended:
            return
        self.integrand.free()
        self.integrand = None
        self._ended = True

    @classmethod
    def from_state(cls, state: 'Integrator.IntegratorState', integrand: MPIntegrand) -> 'Integrator':
        # if type(cls) is Integrator:
        match type(state):
            case Integrator.IntegratorState:
                cls = NaiveIntegrator
            case VegasIntegrator.VegasState:
                cls = VegasIntegrator
            case HavanaIntegrator.HavanaState:
                cls = HavanaIntegrator
            case MadnisIntegrator.MadnisState:
                cls = MadnisIntegrator
            case _:
                raise ValueError("Unknown state type for integrator import.")

        instance = cls.__new__(cls)
        instance.__dict__.update(state.obj_dict)
        instance.integrand = integrand
        instance.import_state(state)

        return instance

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
            r = continuous[:, [i]]
            indices[:, i] = np.sum(cdf < r, axis=1, dtype=np.uint64)
            prob = prob * \
                np.take_along_axis(unnorm_probs, indices[:, [i]], axis=1) / norm

        return indices, (1/prob)

    def _get_samples(self, n_points: int, *args, **kwargs) -> LayerData:
        layer_input = self.init_layer_data(n_points)
        layer_input.continuous = self.rng.random((n_points, self.continuous_dim))
        discrete = self.rng.uniform(
            size=(n_points, len(self.discrete_dims)))
        layer_input.discrete, layer_input.wgt = self._cont_to_discr(discrete)
        layer_input.update(self.IDENTIFIER)

        return layer_input

    def _probe_prob(self, discrete: NDArray, continuous: NDArray | None = None) -> NDArray:
        return np.ones((len(discrete),), dtype=self.dtype)

    def _import_state(self, state: 'Integrator.IntegratorState'):
        pass

    def _export_state(self) -> 'Integrator.IntegratorState':
        return Integrator.IntegratorState(
            obj_dict=self.__dict__.copy(),
            rng_state=deepcopy(self.rng.bit_generator.state))

    @staticmethod
    def from_kwargs(
            graph_properties: GraphProperties | List[GraphProperties],
            parameterisation_kwargs: List[Dict[str, Any]],
            integrand_kwargs: Dict[str, Any],
            integrator_kwargs: Dict[str, Any],) -> 'Integrator':

        integrand = MPIntegrand(
            graph_properties=graph_properties,
            param_kwargs=parameterisation_kwargs,
            integrand_kwargs=integrand_kwargs,
        )

        integrator_type: str | None = integrator_kwargs.pop('integrator_type', None)

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

        return Integrator.from_kwargs(
            graph_properties=Parser.get_graph_properties(),
            parameterisation_kwargs=Parser.get_parameterisation_kwargs(),
            integrand_kwargs=Parser.get_integrand_kwargs(),
            integrator_kwargs=Parser.get_integrator_kwargs()
        )

    def _get_info(self) -> Dict[str, Any]:
        info = {
            "Integrator type": self.IDENTIFIER,
            "Training phase": self.integrand.training_phase,
            "Input dimension": self.input_dim,
            "Continuous dimension": self.continuous_dim,
            "Discrete dimension": self.discrete_dims,
            "Random seed": self.seed,
            "Integrand Cores": self.integrand.n_cores,
        }
        return info

    @staticmethod
    def _default_callback(status: 'Integrator.TrainingStatus') -> None:
        shell_print(
            f"itn {status.step}: {error_fmter(status.acc.training_result.mean, status.acc.training_result.error)}, RSD={status.acc.training_result.rsd:.3f}")

    @dataclass(kw_only=True)
    class IntegratorState:
        obj_dict: Dict[str, Any]
        rng_state: Dict

    @dataclass
    class TrainingStatus:
        step: int
        total_samples: int
        acc: TrainingAccumulator


class NaiveIntegrator(Integrator):
    IDENTIFIER = 'naive sampler'

    def __init__(self,
                 integrand: MPIntegrand,
                 **kwargs,):
        super().__init__(integrand=integrand, **kwargs)


class VegasIntegrator(Integrator):
    IDENTIFIER = 'vegas sampler'

    def __init__(self,
                 integrand: MPIntegrand,
                 bins: int = 500,
                 alpha: float = 0.7,
                 **kwargs):
        super().__init__(integrand=integrand, **kwargs)
        self.bins = bins
        self.adaptive_map = vegas.AdaptiveMap(grid=self.input_dim*[[0, 1],], ninc=bins)
        self.alpha = alpha

    @_block_if_ended
    def train(self, nitn: int = 10, batch_size: int = 1000,
              callback: Callable[['VegasIntegrator.TrainingStatus'], None] | None = None,
              alpha: float | None = None) -> str:
        callback = callback or self._default_callback
        if alpha is None:
            alpha = self.alpha
        shell_print(f"Training Vegas for {nitn} iterations à {batch_size} samples with alpha={alpha}:")
        report = []
        for itn in range(nitn):
            layer_input, r = self.get_samples(batch_size, return_r=True)
            acc: TrainingAccumulator = self.integrand.eval_integrand(layer_input, 'training')
            self.adaptive_map.add_training_data(r, acc.training_data.astype(np.float64))
            self.adaptive_map.adapt(alpha=alpha)
            self.step += 1
            self.total_training_samples += batch_size
            status = Integrator.TrainingStatus(
                step=self.step,
                total_samples=self.total_training_samples,
                acc=acc
            )
            callback(status)
            report.append(f"itn {itn+1}: {error_fmter(acc.training_result.mean, acc.training_result.error)}")

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

    def _probe_prob(self, discrete: NDArray, continuous: NDArray | None = None) -> NDArray:
        prob_disc = np.ones((discrete.shape[0],), dtype=np.float64)
        # For the discrete part, basically invert _cont_to_disc
        for i in range(self.num_discrete_dims):
            disc_dim = self.discrete_dims[i]
            unnorm_probs = self.integrand.discrete_prior_prob_function(
                discrete[:, :i], i)
            cdf = np.cumsum(unnorm_probs, axis=1)
            norm = cdf[:, -1]
            cdf = cdf / norm[:, None]
            # Need to find all intersections of the discrete cdf with the vegas linear interpolation
            g_disc = np.array(self.adaptive_map.grid[self.continuous_dim - 1 + i])
            disc_bins = np.zeros((discrete.shape[0], disc_dim+1), dtype=unnorm_probs.dtype)
            for j in range(disc_dim):
                y_intersection = cdf[:, j]
                x_intersection = np.interp(y_intersection, np.linspace(0, 1, len(g_disc)), g_disc)
                disc_bins[:, j+1] = x_intersection
            probs_proposed = np.diff(disc_bins, axis=1)
            prob_disc *= np.take_along_axis(probs_proposed, discrete[:, [i]]).ravel()

        if continuous is None:
            return prob_disc

        # For the continuous part, we need to go back to y-space (input space) and get the jacobian on y
        x_all = np.hstack([continuous, discrete])
        y_all = np.empty_like(x_all)
        _j = np.empty((x_all.shape[0],), dtype=np.float64)
        self.adaptive_map.invmap(x_all, y_all, _j)
        jac_no_disc = np.prod(self.adaptive_map.jac1d(y_all)[:, :self.continuous_dim], axis=1)

        return prob_disc / jac_no_disc

    def _export_state(self) -> 'VegasIntegrator.VegasState':
        obj_dict = self.__dict__.copy()
        del obj_dict['adaptive_map']
        return VegasIntegrator.VegasState(
            obj_dict=obj_dict,
            rng_state=deepcopy(self.rng.bit_generator.state),
            grid=deepcopy(self.adaptive_map.extract_grid()),
        )

    def _import_state(self, state: 'VegasIntegrator.VegasState'):
        if not isinstance(state, VegasIntegrator.VegasState):
            raise ValueError("State for VegasIntegrator must be of type VegasState.")
        super()._import_state(state)
        self.adaptive_map = vegas.AdaptiveMap(grid=state.grid)
        if self.adaptive_map.dim != self.input_dim:
            raise ValueError("Imported Vegas state does not match input dimension of integrand.")

    def _get_info(self) -> Dict[str, Any]:
        info = super()._get_info()
        info["Bins"] = self.bins
        info["Alpha"] = self.alpha
        return info

    @dataclass(kw_only=True)
    class VegasState(Integrator.IntegratorState):
        grid: List[List[float]]

    @dataclass
    class TrainingStatus:
        iteration: int
        total_samples: int


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
        elif self._discrete_prod < 1000 and not use_uniform:
            def build_nested(dims: List[int], idx: int = 0) -> NumericalIntegrator:
                if idx == len(dims):
                    return NumericalIntegrator.continuous(
                        self.continuous_dim, n_continuous_bins)
                return NumericalIntegrator.discrete(
                    [build_nested(dims, idx+1) for _ in range(dims[idx])],
                    max_prob_ratio,
                )
            self.havana = build_nested(self.discrete_dims)
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
    def train(self, nitn: int = 10, batch_size: int = 1000,
              callback: Callable[[Integrator.TrainingStatus], None] | None = None,
              ) -> str:
        callback = callback or self._default_callback
        report = [
            f"Training Havana for {nitn} iterations à {batch_size} samples:"]
        n_digits = len(str(nitn))
        for itn in range(nitn):
            samples, layer_input = self.get_samples(batch_size, True, True)
            acc: TrainingAccumulator = self.integrand.eval_integrand(layer_input, 'training')
            self.havana.add_training_samples(samples, acc.training_data.tolist())
            res, err, chi_sq = self.havana.update(
                self.discrete_learning_rate, self.continuous_learning_rate)
            self.step += 1
            self.total_training_samples += batch_size
            status = Integrator.TrainingStatus(
                step=self.step,
                total_samples=self.total_training_samples,
                acc=acc
            )
            callback(status)
            report.append(
                f"It {itn+1:0>{n_digits}}: {error_fmter(res, err)}, χ²={chi_sq:.3f}")

        return "\n".join(f"{line}" for line in report)

    def _get_samples(self, n_points: int, return_samples: bool = False, training: bool = False,
                     ) -> Tuple[List[Sample], LayerData] | LayerData:
        layer_input = self.init_layer_data(n_points)
        samples = self.havana.sample(n_points, self.symbolica_rng)

        continuous = np.empty(
            (n_points, self.continuous_dim), dtype=self.integrand.dtype)
        for i, sample in enumerate(samples):
            continuous[i] = sample.c

        if self.num_discrete_dims > 0:
            discrete = np.empty(
                (n_points, self.num_discrete_dims), dtype=np.uint64)
            for i, sample in enumerate(samples):
                discrete[i] = sample.d
        else:
            discrete = np.zeros((n_points, 0), dtype=np.uint64)

        layer_input.continuous = continuous
        layer_input.discrete = discrete
        total_wgt = self.integrand.apply_prior_to_discrete(discrete)

        if not training:
            total_wgt *= np.array([s.weights[0] for s in samples])

        layer_input.wgt = total_wgt
        layer_input.update(self.IDENTIFIER)

        if return_samples:
            return samples, layer_input

        return layer_input

    def _probe_prob(self, discrete: NDArray, continuous: NDArray | None = None) -> NDArray:
        from symbolica import Probe
        n_samples = len(discrete)
        if continuous is None:
            continuous = np.zeros((n_samples, 0))
        prob = 1. / np.array([
            self.havana.probe(Probe.discrete(d, c)) for d, c in zip(discrete, continuous)
        ])
        if discrete.shape[1] > 0:
            prob *= self.integrand.apply_prior_to_discrete(discrete)

        return prob

    def _export_state(self) -> 'HavanaIntegrator.HavanaState':
        obj_dict = self.__dict__.copy()
        obj_dict.pop('havana', None)
        obj_dict.pop('symbolica_rng', None)

        return HavanaIntegrator.HavanaState(
            obj_dict=obj_dict,
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
        info["Bins"] = self.bins
        info["Max Probability Ratio"] = self.max_prob_ratio
        info["Using Uniform Discrete Grid"] = self.uniform_disc_grid
        info["Discrete Learning Rate"] = self.discrete_learning_rate
        info["Continuous Learning Rate"] = self.continuous_learning_rate
        return info


class MadnisIntegrator(Integrator):
    IDENTIFIER = 'madnis sampler'

    def __init__(
            self,
            integrand: MPIntegrand,
            batch_size: int = 1000,
            learning_rate: float = 1e-3,
            use_scheduler: bool = True,
            scheduler_type: Literal["cosineannealing"] | List[str] | None = None,
            loss_type: Literal["test", "variance", "variance_softclip",
                               "kl_divergence", "kl_divergence_softclip"] = "kl_divergence",
            discrete_dims_position: Literal["first", "last"] = "first",
            discrete_model: Literal["transformer", "made"] = "transformer",
            use_cwnet: bool = False,
            train_continuous_flow: bool = True,
            train_discrete_flow: bool = True,
            train_cwnet: bool = True,
            max_batch_size: int = 100_000,
            use_gpu: bool = True,
            gpu_id: int = 0,
            pretrain_c_flow: bool = False,
            flow_kwargs: Dict[str, Any] = dict(
                uniform_latent=True,
                permutations="log",
                layers=3,
                units=64,
                bins=10,
                min_bin_width=1e-3,
                min_bin_height=1e-3,
                min_bin_derivative=1e-3,),
            transformer_kwargs: Dict[str, Any] = dict(
                embedding_dim=64,
                feedforward_dim=64,
                heads=4,
                mlp_units=64,
                transformer_layers=2,),
            made_kwargs: Dict[str, Any] = dict(
                layers=3,
                nodes_per_feature=64,
            ),
            cwnet_kwargs: Dict[str, Any] = dict(
                layers=4,
                units=128,
                uniform_channel_ratio=0.1,
                channel_weight_mode="variance",
            ),
            scheduler_kwargs: Dict[str, Any] = dict(),
            loss_kwargs: Dict[str, Any] = dict(),
            pretraining_kwargs: Dict[str, Any] = dict(
                nitn=10,
                neval=10000,
                bins_mult=4,
                alpha=0.7,
            ),
            **kwargs,
    ):
        super().__init__(integrand, **kwargs)
        torch.set_default_dtype(torch.float64)

        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.device = self._get_device()
        self.use_cwnet = use_cwnet and self.num_discrete_dims > 0
        self.uniform_channel_ratio = cwnet_kwargs.pop("uniform_channel_ratio", 0.1) if self.use_cwnet else 0.0
        self.channel_weight_mode = cwnet_kwargs.pop("channel_weight_mode", "variance") if self.use_cwnet else None
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.seed)

        self.use_scheduler = use_scheduler
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs
        self.batch_size = batch_size
        self.discrete_dims_position = discrete_dims_position

        self.loss_type = loss_type
        self.loss_kwargs = loss_kwargs

        self.discrete_model = discrete_model if not self.use_cwnet else "made"
        discrete_flow_kwargs = transformer_kwargs if self.discrete_model == "transformer" else made_kwargs

        madnis_integrand = madnis_integrator.Integrand(
            function=self._madnis_eval,
            input_dim=self.input_dim - int(self.use_cwnet),
            channel_count=self.discrete_dims[0] if self.use_cwnet else None,
            discrete_dims=self.discrete_dims[int(self.use_cwnet):],
            discrete_dims_position=self.discrete_dims_position,
            discrete_prior_prob_function=self._madnis_discrete_prior_prob_function,
            has_channel_weight_prior=self.use_cwnet,
        )
        if self.use_cwnet:
            self.madnis = madnis_integrator.Integrator(
                madnis_integrand,
                device=self.device,
                discrete_flow_kwargs=made_kwargs,
                loss=self._get_loss(),
                batch_size=self.batch_size,
                discrete_model="made",
                learning_rate=learning_rate,
                flow_kwargs=flow_kwargs,
                uniform_channel_ratio=self.uniform_channel_ratio,
                train_channel_weights=self.use_cwnet,
                cwnet_kwargs=cwnet_kwargs,
            )
        else:
            self.madnis = madnis_integrator.Integrator(
                madnis_integrand,
                device=self.device,
                discrete_flow_kwargs=discrete_flow_kwargs,
                loss=self._get_loss(),
                batch_size=self.batch_size,
                discrete_model=discrete_model,
                learning_rate=learning_rate,
                flow_kwargs=flow_kwargs,
            )
        self.max_batch_size = max_batch_size

        if not train_cwnet and self.use_cwnet:
            self.madnis.cwnet.eval()
            for param in self.madnis.cwnet.parameters():
                param.requires_grad = False

        if not train_continuous_flow:
            if hasattr(self.madnis.flow, 'continuous_flow'):
                self.madnis.flow.continuous_flow.eval()
                for param in self.madnis.flow.continuous_flow.parameters():
                    param.requires_grad = False
            else:
                self.madnis.flow.eval()
                for param in self.madnis.flow.parameters():
                    param.requires_grad = False

        if not train_discrete_flow and hasattr(self.madnis.flow, 'discrete_flow'):
            if self.madnis.flow.discrete_flow is not None:
                self.madnis.flow.discrete_flow.eval()
                for param in self.madnis.flow.discrete_flow.parameters():
                    param.requires_grad = False

        if pretrain_c_flow:
            bins = max(int(pretraining_kwargs.get('bins_mult', 4)*flow_kwargs.get('bins', 100)), 200)
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
    def train(self, nitn: int = 10, batch_size: int | None = None,
              callback: Callable[['MadnisIntegrator.TrainingStatus'], None] | None = None) -> str:
        import signal

        if batch_size is not None:
            self.madnis.batch_size = batch_size
        callback = callback or self._default_callback
        if self.use_scheduler:
            self.madnis.scheduler = self._get_scheduler(
                self.madnis.step + nitn, self.scheduler_type)

        interrupted = False

        def handler(sig, frame):
            nonlocal interrupted
            interrupted = True

        old_handler = signal.signal(signal.SIGINT, handler)
        try:
            for _ in range(nitn):
                madnis_status = self.madnis.train_step()
                if interrupted:
                    break
                self.step += 1
                self.total_training_samples += self.madnis.batch_size
                status = self.TrainingStatus(
                    step=self.step,
                    total_samples=self.total_training_samples,
                    madnis_status=madnis_status,
                )
                callback(status)
        finally:
            signal.signal(signal.SIGINT, old_handler)

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
                madnis_batch = self.madnis._get_samples(
                    n,
                    uniform_channel_ratio=self.uniform_channel_ratio,
                    train=False,
                    channel_weight_mode=self.channel_weight_mode,
                    channel=None,
                    evaluate_integrand=False,
                )
            discrete[n_eval:n_eval+n, :], continuous[n_eval:n_eval+n, :] = self._madnis_output_to_disc_cont(
                madnis_batch.x, madnis_batch.channels)
            wgt[n_eval:n_eval+n, :] = 1 / madnis_batch.q_sample.numpy(force=True).reshape(-1, 1)

            if madnis_batch.channels is not None:
                # We multiply by the number of channels to compensate for the CWNET normalization
                wgt[n_eval:n_eval+n, :] *= torch.gather(
                    self.madnis._get_alphas(madnis_batch), index=madnis_batch.channels[:, None], dim=1
                )[:, 0].numpy(force=True).reshape(-1, 1) * self.discrete_dims[0]
                # Next, weigh by N_i / N to restore the actual multichanneling alphas
                channels = madnis_batch.channels.numpy(force=True)
                channel_counts = np.bincount(channels, minlength=self.discrete_dims[0])
                wgt[n_eval:n_eval+n, :] *= (n / channel_counts[channels]).reshape(-1, 1)

            n_eval += n

        layer_input.continuous = continuous
        layer_input.discrete = discrete
        layer_input.wgt = wgt
        layer_input.update(self.IDENTIFIER)

        return layer_input

    def _madnis_eval(self, x_all: Tensor, channel: Tensor | None = None) -> Tensor:
        layer_input = self.init_layer_data(x_all.shape[0])
        layer_input.discrete, layer_input.continuous = self._madnis_output_to_disc_cont(x_all, channel)
        layer_input.update(self.IDENTIFIER)

        acc: TrainingAccumulator = self.integrand.eval_integrand(layer_input, 'training')
        torch_output = torch.from_numpy(
            acc.training_data.astype(np.float64)).to(self.device)

        if self.madnis.integrand.has_channel_weight_prior:
            disc_prior = self.integrand.discrete_prior_prob_function(
                np.zeros((x_all.shape[0], 0), dtype=np.uint64), 0)
            disc_prior /= np.sum(disc_prior, axis=1, keepdims=True) / self.discrete_dims[0]
            prior_wgt = np.take_along_axis(disc_prior, layer_input.discrete[:, [0]])
            torch_prior = torch.from_numpy(prior_wgt.astype(np.float64)).to(self.device)
            return torch_output, torch_prior

        return torch_output

    def _probe_prob(self, discrete: NDArray, continuous: NDArray | None = None) -> NDArray:
        n_samples = discrete.shape[0]
        if continuous is None:
            continuous = np.zeros((n_samples, 0), dtype=discrete.dtype)
        if self.use_cwnet:
            if continuous.size == 0:
                ch_ctrb = self.madnis._get_channel_contributions(False, self.channel_weight_mode).numpy(force=True)
                return ch_ctrb[discrete[:, 0]]
            channel = torch.from_numpy(discrete[:, 0]).to(self.device, torch.int64)
            discrete = discrete[:, 1:]
        else:
            channel = None
        if self.madnis.integrand.discrete_dims_position == "first":
            x_all = np.hstack([discrete, continuous])
        elif self.madnis.integrand.discrete_dims_position == "last":
            x_all = np.hstack([continuous, discrete])
        prob = np.empty((n_samples,), dtype=np.float64)
        x_all = torch.as_tensor(
            x_all.astype(np.float64),
            device=self.madnis.dummy.device,
            dtype=self.madnis.dummy.dtype if continuous.shape[1] > 0 else torch.int64,
        )

        n_eval = 0
        while n_eval < n_samples:
            n = min(self.max_batch_size, n_samples - n_eval)
            with torch.no_grad():
                if continuous.shape[1] > 0:
                    prob[n_eval:n_eval+n] = self.madnis.flow.prob(
                        x_all[n_eval:n_eval+n, :],
                        channel=None if channel is None else channel[n_eval:n_eval+n]).numpy(force=True).reshape(-1)
                else:
                    prob[n_eval:n_eval+n] = self.madnis.flow.discrete_flow.prob(
                        x_all[n_eval:n_eval+n, :],
                        channel=None if channel is None else channel[n_eval:n_eval+n]).numpy(force=True).reshape(-1)
            n_eval += n

        return prob

    def _madnis_discrete_prior_prob_function(self, indices: Tensor, dim: int = 0) -> Tensor:
        indices = indices.numpy(force=True).astype(np.uint64)
        if self.use_cwnet:
            indices = np.hstack([np.zeros((indices.shape[0], 1), dtype=np.uint64), indices])
            dim += 1
        numpy_output = self.integrand.discrete_prior_prob_function(indices, dim)

        return torch.from_numpy(numpy_output.astype(np.float64)).to(self.device)

    def _madnis_output_to_disc_cont(self, x_all: Tensor, channel: Tensor | None = None) -> Tuple[NDArray, NDArray]:
        n_disc = self.num_discrete_dims - int(self.use_cwnet)
        if self.discrete_dims_position == "first":
            discrete = x_all[:, :n_disc].numpy(force=True)
            continuous = x_all[:, n_disc:].numpy(force=True)
        else:
            discrete = x_all[:, -n_disc:].numpy(force=True)
            continuous = x_all[:, :-n_disc].numpy(force=True)
        if channel is not None:
            discrete = np.hstack([channel.numpy(force=True).reshape(-1, 1), discrete])
        return discrete, continuous

    @staticmethod
    def _default_callback(status: 'MadnisIntegrator.TrainingStatus') -> None:
        if (status.step + 1) % 10 == 0:
            shell_print(f"Step {status.step+1}: Loss={status.madnis_status.loss} ")

    def _export_state(self) -> 'MadnisIntegrator.MadnisState':
        obj_dict = self.__dict__.copy()
        obj_dict.pop('madnis', None)
        buffer = io.BytesIO()
        tmp_integrand = self.madnis.integrand
        tmp_loss = self.madnis.loss
        tmp_ch_remap = (
            None
            if self.madnis.group_channels and not self.madnis.group_channels_uniform
            else tmp_integrand.remap_channels
        )
        try:
            self.madnis.integrand = None
            self.madnis.loss = None
            self.madnis.flow.prior_prob_function = None
            if hasattr(self.madnis.flow, 'channel_remap_function'):
                self.madnis.flow.channel_remap_function = None
            if hasattr(self.madnis.flow, 'continuous_flow') and self.madnis.flow.continuous_flow is not None:
                self.madnis.flow.continuous_flow.channel_remap_function = None
            if hasattr(self.madnis.flow, 'discrete_flow') and self.madnis.flow.discrete_flow is not None:
                self.madnis.flow.discrete_flow.prior_prob_function = None
                if hasattr(self.madnis.flow.discrete_flow, 'channel_remap_function'):
                    self.madnis.flow.discrete_flow.channel_remap_function = None

            # from glnis.utils.helpers import Colour

            # def try_nested(obj, key="Madnis"):
            #     for sub_key, sub_obj in (obj.items() if hasattr(obj, 'items') else obj.__dict__.items()):
            #         try:
            #             torch.save(sub_obj, buffer)
            #         except Exception as e:
            #             print(
            #                 f"Error occurred while saving {Colour.CYAN}{key}{Colour.END}.{Colour.RED}{sub_key}{Colour.END}: {e}")
            #             try_nested(sub_obj, f"{key}.{sub_key}")
            # try_nested(self.madnis)
            torch.save(self.madnis, buffer)
            madnis_blob = buffer.getvalue()
        finally:
            self.madnis.integrand = tmp_integrand
            self.madnis.loss = tmp_loss
            self.madnis.flow.prior_prob_function = self._madnis_discrete_prior_prob_function
            if hasattr(self.madnis.flow, 'channel_remap_function'):
                self.madnis.flow.channel_remap_function = tmp_ch_remap
            if hasattr(self.madnis.flow, 'continuous_flow') and self.madnis.flow.continuous_flow is not None:
                self.madnis.flow.continuous_flow.channel_remap_function = tmp_ch_remap
            if hasattr(self.madnis.flow, 'discrete_flow') and self.madnis.flow.discrete_flow is not None:
                self.madnis.flow.discrete_flow.prior_prob_function = self._madnis_discrete_prior_prob_function
                if hasattr(self.madnis.flow.discrete_flow, 'channel_remap_function'):
                    self.madnis.flow.discrete_flow.channel_remap_function = tmp_ch_remap
        return self.MadnisState(
            obj_dict=obj_dict,
            rng_state=deepcopy(self.rng.bit_generator.state),
            torch_cpu_rng_state=torch.get_rng_state(),
            torch_gpu_rng_state=torch.cuda.get_rng_state() if self.device.type == 'cuda' else None,
            madnis_blob=madnis_blob,
        )

    def _import_state(self, state: 'MadnisIntegrator.MadnisState'):
        if not isinstance(state, MadnisIntegrator.MadnisState):
            raise ValueError("Invalid state type for MadnisIntegrator.")
        super()._import_state(state)
        self.device = self._get_device()
        buffer = io.BytesIO(state.madnis_blob)
        self.madnis: madnis_integrator.Integrator = torch.load(buffer, map_location=self.device, weights_only=False)
        self.madnis.integrand = madnis_integrator.Integrand(
            function=self._madnis_eval,
            input_dim=self.input_dim - int(self.use_cwnet),
            channel_count=self.discrete_dims[0] if self.use_cwnet else None,
            discrete_dims=self.discrete_dims[int(self.use_cwnet):],
            discrete_dims_position=self.discrete_dims_position,
            discrete_prior_prob_function=self._madnis_discrete_prior_prob_function,
            has_channel_weight_prior=self.use_cwnet,
        )
        ch_remap = (
            None
            if self.madnis.group_channels and not self.madnis.group_channels_uniform
            else self.madnis.integrand.remap_channels
        )
        self.madnis.loss = self._get_loss()
        if hasattr(self.madnis.flow, 'channel_remap_function'):
            self.madnis.flow.channel_remap_function = ch_remap
        if hasattr(self.madnis.flow, 'continuous_flow') and self.madnis.flow.continuous_flow is not None:
            self.madnis.flow.continuous_flow.channel_remap_function = ch_remap
        if hasattr(self.madnis.flow, 'discrete_flow') and self.madnis.flow.discrete_flow is not None:
            self.madnis.flow.discrete_flow.prior_prob_function = self._madnis_discrete_prior_prob_function
            if hasattr(self.madnis.flow.discrete_flow, 'channel_remap_function'):
                self.madnis.flow.discrete_flow.channel_remap_function = ch_remap
        if self.device.type == 'cuda':
            if state.torch_gpu_rng_state is None:
                shell_print("Warning: No GPU RNG state found in the saved state. GPU computations may not be reproducible.")
            torch.cuda.set_rng_state(state.torch_gpu_rng_state)
        else:
            if state.torch_gpu_rng_state is not None:
                shell_print(
                    "Warning: GPU RNG state found in the saved state but current device is CPU. Computations may not be reproducible")
        torch.set_rng_state(state.torch_cpu_rng_state)

    def _get_info(self) -> Dict[str, Any]:
        info = super()._get_info()
        info["Scheduler"] = self.scheduler_type
        info["Device"] = str(self.device)
        if self.num_discrete_dims > int(self.use_cwnet):
            info["Discrete Model"] = self.discrete_model
            trainable_disc_flow = sum(p.numel() for p in self.madnis.flow.discrete_flow.parameters() if p.requires_grad)
            total_disc_flow = sum(p.numel() for p in self.madnis.flow.discrete_flow.parameters())
            info["Discrete flow trainable parameters"] = trainable_disc_flow
            info["Discrete flow total parameters"] = total_disc_flow

            trainable_cont_flow = sum(p.numel()
                                      for p in self.madnis.flow.continuous_flow.parameters() if p.requires_grad)
            total_cont_flow = sum(p.numel() for p in self.madnis.flow.continuous_flow.parameters())
            info["Continuous flow trainable parameters"] = trainable_cont_flow
            info["Continuous flow total parameters"] = total_cont_flow

        trainable_flow = sum(p.numel() for p in self.madnis.flow.parameters() if p.requires_grad)
        total_flow = sum(p.numel() for p in self.madnis.flow.parameters())
        info["Flow trainable parameters"] = trainable_flow
        info["Flow total parameters"] = total_flow

        if self.madnis.cwnet is not None:
            trainable_cwnet = sum(p.numel() for p in self.madnis.cwnet.parameters() if p.requires_grad)
            total_cwnet = sum(p.numel() for p in self.madnis.cwnet.parameters())
            info["CWNet trainable parameters"] = trainable_cwnet
            info["CWNet total parameters"] = total_cwnet

        return info

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

    def _get_device(self) -> torch.device:
        if not self.use_gpu or not torch.cuda.is_available():
            return torch.device('cpu')

        n_dvc = torch.cuda.device_count()
        gpu_id = min(self.gpu_id, n_dvc - 1)
        reordered_list = list(range(n_dvc))
        reordered_list.pop(gpu_id)
        reordered_list.insert(0, gpu_id)
        for i in reordered_list:
            major, minor = torch.cuda.get_device_capability(i)
            if (7, 0) <= (major, minor) < (12, 0):
                torch.cuda.set_device(i)
                return torch.device(f'cuda:{i}')
        shell_print("CUDA devices found but none are compatible. Using CPU.")
        return torch.device('cpu')

    def _get_loss(self):
        from glnis.core import losses
        loss = None
        match self.loss_type.lower():
            case "variance":
                loss = madnis_integrator.losses.stratified_variance
            case "variance_softclip":
                loss = madnis_integrator.losses.stratified_variance_softclip
            case "kl_divergence":
                loss = madnis_integrator.losses.kl_divergence
            case "kl_divergence_softclip":
                loss = madnis_integrator.losses.kl_divergence_softclip
            case "rkl_divergence":
                loss = madnis_integrator.losses.rkl_divergence
            case "test":
                loss = losses.test()
            case _:
                pass

        if loss is None:
            return None

        def loss_with_kwargs(*args, **kwargs):
            return loss(*args, **kwargs, **self.loss_kwargs)

        return loss_with_kwargs

    @dataclass(kw_only=True)
    class MadnisState(Integrator.IntegratorState):
        torch_cpu_rng_state: Tensor
        torch_gpu_rng_state: Tensor | None
        madnis_blob: bytes

        def __setstate__(self, state_dict):
            legacy_rng_state = state_dict.pop('torch_rng_state', None)
            if legacy_rng_state is not None:
                state_dict['torch_cpu_rng_state'] = legacy_rng_state
                state_dict['torch_gpu_rng_state'] = None
            self.__dict__.update(state_dict)

    @dataclass
    class TrainingStatus:
        step: int
        total_samples: int
        madnis_status: madnis_integrator.TrainingStatus
