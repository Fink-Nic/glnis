# type: ignore
import vegas
import numpy as np
import torch
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
from glnis.utils.helpers import shell_print


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

    @abstractmethod
    def get_samples(self, n_points: int) -> LayerData:
        """Returns a LayerData object containing the samples to be fed into the integrand."""
        pass

    @abstractmethod
    def export_state(self) -> Any:
        """Exports the state of the integrator, e.g. for checkpointing or analysis."""
        pass

    @abstractmethod
    def import_state(self, state: Any):
        """
        Imports the state of the integrator from a previous export.
        Upon importing, the integrand must be set up to match the imported state.
        """
        pass

    def integrate(self,
                  n_samples: int,
                  n_start: int = 1_000_000,
                  n_increase: int = 0,
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
                shell_print(f"Evaluated {n_eval} / {n_samples} samples using {self.IDENTIFIER}...")
                shell_print(accumulator.str_report())
            n_curr_iter += n_increase

        return accumulator

    def train(self, nitn: int = 10, batch_size: int = 10000, ):
        print(f"Training not available for Integrator {self.IDENTIFIER}.")

    def _cont_to_discr(self, continuous: NDArray
                       ) -> Tuple[NDArray, NDArray]:
        n = len(continuous)
        if not continuous.shape[1] == self.num_discrete_dims:
            raise ValueError(
                "Shape of sampler output does not match discrete dims of stack.")

        indices = np.zeros((n, 0), dtype=np.uint64)
        prob = np.ones((n,))
        for i in range(self.num_discrete_dims):
            unnorm_probs = self.integrand.discrete_prior_prob_function(
                indices[:, :i], i)
            cdf = np.cumsum(unnorm_probs, axis=1)
            norm = cdf[:, -1]
            cdf = cdf / norm[:, None]
            r = np.random.random_sample((n, 1))
            samples = np.sum(cdf < r, axis=1, dtype=np.uint64).reshape(-1, 1)
            indices = np.hstack((indices, samples))
            prob = prob * \
                np.take_along_axis(unnorm_probs, samples, axis=1)[:, 0] / norm

        return indices, (1/prob).reshape(-1, 1)

    def init_layer_data(self, n_points: int) -> LayerData:
        return LayerData(
            n_points,
            n_mom=3*self.integrand.graph_properties.n_loops,
            n_cont=self.continuous_dim,
            n_disc=self.num_discrete_dims,
            dtype=self.dtype,
        )

    @staticmethod
    def from_dicts(
            graph_properties: GraphProperties,
            parameterisation_kwargs: List[Dict[str, Any]],
            integrand_kwargs: Dict[str, Any],
            integrator_kwargs: Dict[str, Any],) -> 'Integrator':

        n_cores = integrand_kwargs.pop('n_cores')
        verbose = integrand_kwargs.pop('verbose')
        integrator_type = integrator_kwargs.pop('integrator_type')

        integrand = MPIntegrand(
            graph_properties=graph_properties,
            param_kwargs=parameterisation_kwargs,
            integrand_kwargs=integrand_kwargs,
            n_cores=n_cores,
            verbose=verbose,
        )

        match integrator_type.lower():
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
    def from_settings_file(settings_path: str) -> 'Integrator':
        Parser = SettingsParser(settings_path)
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


class NaiveIntegrator(Integrator):
    IDENTIFIER = 'naive sampler'

    def __init__(self,
                 integrand: MPIntegrand,
                 **kwargs,):
        super().__init__(integrand=integrand, **kwargs)
        self.rng = np.random.default_rng(self.seed)

    def get_samples(self, n_points: int) -> LayerData:
        layer_input = self.init_layer_data(n_points)
        layer_input.continuous = self.rng.random((n_points, self.continuous_dim))
        discrete = self.rng.uniform(
            size=(n_points, len(self.discrete_dims)))
        layer_input.discrete, layer_input.wgt = self._cont_to_discr(discrete)
        layer_input.update(self.IDENTIFIER)

        return layer_input

    def export_state(self) -> 'NaiveIntegrator.NaiveState':
        return NaiveIntegrator.NaiveState(rng=deepcopy(self.rng))

    def import_state(self, state: 'NaiveIntegrator.NaiveState'):
        if not isinstance(state, NaiveIntegrator.NaiveState):
            raise ValueError("State for NaiveIntegrator must be of type NaiveState.")
        self.rng = np.random.default_rng(deepcopy(state.rng))

    @dataclass
    class NaiveState:
        rng: np.random._generator.Generator


class VegasIntegrator(Integrator):
    IDENTIFIER = 'vegas sampler'

    def __init__(self,
                 integrand: MPIntegrand,
                 vegas_init_kwargs: Dict[str, Any] = dict(),
                 **kwargs):
        super().__init__(integrand=integrand, **kwargs)
        self.rng = np.random.default_rng(self.seed)
        self.vegas_init_kwargs = vegas_init_kwargs
        self.vegas = vegas.Integrator(self.input_dim*[[0, 1],], **self.vegas_init_kwargs)
        self.vegas.ran_array_generator = self.rng.random

    def train(self, nitn: int = 10, batch_size: int = 10000, **vegas_kwargs) -> vegas._vegas.RAvg:
        return self.vegas(self._vegas_wrapper, nitn=nitn, neval=batch_size, **vegas_kwargs)

    def get_samples(self, n_points: int) -> LayerData:
        # Just need this for the timestamp, since vegas may decide on a different sample size.
        dummy_input = self.init_layer_data(1)
        sampling_wgt, samples = self.vegas.sample(n_points, mode="lbatch")
        sampling_wgt = sampling_wgt.reshape(-1, 1)
        n_points = len(sampling_wgt)
        layer_input = self.init_layer_data(n_points)
        layer_input._timestamp = dummy_input._timestamp
        layer_input.continuous = samples[:, :self.continuous_dim]
        layer_input.discrete, disc_wgt = self._cont_to_discr(
            samples[:, self.continuous_dim:])
        layer_input.wgt *= disc_wgt * sampling_wgt * n_points
        layer_input.update(self.IDENTIFIER)

        return layer_input

    @vegas.lbatchintegrand
    def _vegas_wrapper(self, x: NDArray) -> NDArray:
        layer_input = self.init_layer_data(len(x))
        layer_input.continuous = x[:, :self.continuous_dim]
        layer_input.discrete, layer_input.wgt = self._cont_to_discr(
            x[:, self.continuous_dim:])
        layer_input.update(self.IDENTIFIER)
        accumulated_result: TrainingAccumulator = self.integrand.eval_integrand(
            layer_input, 'training')

        return accumulated_result.training_data.training_result[0]

    def export_state(self) -> 'VegasIntegrator.VegasState':
        return VegasIntegrator.VegasState(integrator=deepcopy(self.vegas),
                                          rng=deepcopy(self.rng))

    def import_state(self, state: 'VegasIntegrator.VegasState'):
        if not isinstance(state, VegasIntegrator.VegasState):
            raise ValueError("State for VegasIntegrator must be of type VegasState.")
        self.rng = deepcopy(state.rng)
        self.vegas = state.integrator
        if self.vegas.map.dim != self.input_dim:
            raise ValueError("Imported Vegas state does not match input dimension of integrand.")
        self.vegas.ran_array_generator = self.rng.random

    @dataclass
    class VegasState:
        integrator: vegas.Integrator
        rng: np.random._generator.Generator


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
        elif self.num_discrete_dims == 1 and not use_uniform:
            self.havana = NumericalIntegrator.discrete(
                [NumericalIntegrator.continuous(
                    self.continuous_dim) for _ in range(self.discrete_dims[0])],
                max_prob_ratio,
            )
        else:
            self.havana = NumericalIntegrator.uniform(
                self.discrete_dims,
                NumericalIntegrator.continuous(
                    self.continuous_dim, n_continuous_bins)
            )

        self.stream_id = stream_id
        self.rng = RandomNumberGenerator(self.seed, self.stream_id)
        self.discrete_learning_rate = discrete_learning_rate
        self.continuous_learning_rate = continuous_learning_rate

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

        return "\n".join(f"| > {line}" for line in report)

    def get_samples(self, n_points: int, return_samples: bool = False, training: bool = False,
                    ) -> Tuple[List[Sample], LayerData] | LayerData:
        layer_input = self.init_layer_data(n_points)
        samples = self.havana.sample(n_points, self.rng)

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
                print(f"Error computing discrete weight for dimension {i}: {e}")
                print(f"Discrete samples for this dimension: {discrete[:, i]}")
                raise e

        if training:
            total_wgt /= np.array(self.discrete_dims).prod()
        else:
            sample_weights = np.array([s.weights for s in samples])
            sample_weights = sample_weights[:, -1]
            # sample_disc_weights = sample_weights[:, :-1].prod(axis=1)
            # print(f"{sample_disc_weights.sum():.3e} (sum of discrete part of current sample weights)")
            # print(f"{sample_disc_weights.mean():.3e} +- {sample_disc_weights.std():.3e} (mean weight and stddev of discrete part of current samples)")
            # print(f"{sample_weights.mean():.3e} +- {sample_weights.std():.3e} (mean weight and stddev of current samples)")
            # print(f"{total_wgt.mean():.3e} +- {total_wgt.std():.3e} (mean weight and stddev of current total weights before sample weights)")
            total_wgt *= sample_weights  # * sample_disc_weights

        layer_input.wgt = total_wgt
        layer_input.update(self.IDENTIFIER)

        if return_samples:
            return samples, layer_input

        return layer_input

    def export_state(self) -> 'HavanaIntegrator.HavanaState':
        return HavanaIntegrator.HavanaState(grid=deepcopy(self.havana.export_grid()),
                                            seed=self.seed,
                                            stream_id=self.stream_id,
                                            # rng=deepcopy(self.rng),
                                            )

    def import_state(self, state: 'HavanaIntegrator.HavanaState'):
        if not isinstance(state, HavanaIntegrator.HavanaState):
            raise ValueError("State for HavanaIntegrator must be of type HavanaState.")
        self.havana.merge(NumericalIntegrator.import_grid(state.grid))
        self.seed = state.seed
        self.stream_id = state.stream_id
        self.rng = RandomNumberGenerator(self.seed, self.stream_id)
        # self.rng = state.rng

    @dataclass
    class HavanaState:
        grid: bytes
        seed: int
        stream_id: int
        # rng: RandomNumberGenerator


class MadnisIntegrator(Integrator):
    IDENTIFIER = 'madnis sampler'

    def __init__(self,
                 integrand: MPIntegrand,
                 batch_size: int = 1024,
                 learning_rate: float = 1e-3,
                 use_scheduler: bool = True,
                 scheduler_type: Literal["cosineannealing"] | None = None,
                 scheduler_kwargs: Dict[str, Any] = dict(),
                 loss_type: Literal["variance", "variance_softclip",
                                    "kl_divergence", "kl_divergence_softclip"] = "kl_divergence",
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
                 callback: Callable[[object], None] | None = None,
                 **kwargs,):
        super().__init__(integrand, **kwargs)
        import torch
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(self.seed)

        self.device = torch.device('cpu')  # default

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                if (7, 0) <= (major, minor) < (12, 0):
                    self.device = torch.device(f'cuda:{i}')
                    print(
                        f"Using CUDA device {i}: {torch.cuda.get_device_name(i)} (capability {major}.{minor})")
                    break
            else:
                print("CUDA devices found but none are compatible. Using CPU.")
        else:
            print("No CUDA device found. Using CPU.")

        self.use_scheduler = use_scheduler
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs
        self.batch_size = batch_size
        self.discrete_dims_position = discrete_dims_position

        match loss_type.lower():
            case "variance":
                loss = madnis_integrator.losses.stratified_variance
            case "variance_softclip":
                loss = MadnisIntegrator.stratified_variance_softclip
            case "kl_divergence":
                loss = madnis_integrator.losses.kl_divergence
            case "kl_divergence_softclip":
                loss = MadnisIntegrator.kl_divergence_softclip
            case _:
                loss = None

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
            loss=loss,
            batch_size=self.batch_size,
            discrete_model=discrete_model,
            learning_rate=learning_rate,
            flow_kwargs=flow_kwargs,
        )
        self.callback = self._default_callback if callback is None else callback

    def train(self, nitn: int = 10, _=None):
        if self.use_scheduler:
            self.madnis.scheduler = self._get_scheduler(
                nitn, self.scheduler_type)
        self.madnis.train(nitn, self.callback, True)

    def get_samples(self, n_points: int, batch_size: int = 100_000) -> LayerData:
        layer_input = self.init_layer_data(n_points)
        # LayerData objects return a view of the fields, so we can fill them directly,
        # but this would sidestep data validation, so we fill a copy and then assign it
        continuous = np.empty((n_points, self.continuous_dim), dtype=self.integrand.dtype)
        discrete = np.empty((n_points, self.num_discrete_dims), dtype=np.uint64)
        wgt = np.empty((n_points, 1), dtype=self.integrand.dtype)

        n_eval = 0
        while n_eval < n_points:
            n = min(batch_size, n_points - n_eval)
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
        torch_output = torch.from_numpy(
            weighted_func_val.astype(np.float64)).to(self.device)
        return torch_output

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
            case _:
                return None

    @staticmethod
    def _default_callback(status: madnis_integrator.TrainingStatus) -> None:
        print(f"Step {status.step+1}: Loss={status.loss} ")

    @staticmethod
    def softclip(x: torch.Tensor, threshold: torch.Tensor = 30.0):
        return threshold * torch.arcsinh(x / threshold)

    @staticmethod
    def stratified_variance_softclip(
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor | None = None,
        channels: torch.Tensor | None = None,
        threshold: torch.Tensor = 30.0,
    ):
        """
        Computes the stratified variance as introduced in [2311.01548] for two given sets of
        probabilities, ``f_true`` and ``q_test``. It uses importance sampling with a sampling
        probability specified by ``q_sample``. A soft clipping function is applied to the
        sample weights.

        Args:
            f_true: normalized integrand values
            q_test: estimated function/probability
            q_sample: sampling probability
            channels: channel indices or None in the single-channel case
            threshold: approximate point of transition between linear and logarithmic behavior
        Returns:
            computed stratified variance
        """
        if q_sample is None:
            q_sample = q_test
        if channels is None:
            norm = torch.mean(f_true.detach().abs() / q_sample)
            f_true = MadnisIntegrator.softclip(
                f_true / q_sample / norm, threshold) * q_sample * norm
            abs_integral = torch.mean(f_true.detach().abs() / q_sample)
            return madnis_integrator.losses._variance(f_true, q_test, q_sample) / abs_integral.square()

        stddev_sum = 0
        abs_integral = 0
        for i in channels.unique():
            mask = channels == i
            fi, qti, qsi = f_true[mask], q_test[mask], q_sample[mask]
            norm = torch.mean(fi.detach().abs() / qsi)
            fi = MadnisIntegrator.softclip(
                fi / qsi / norm, threshold) * qsi * norm
            stddev_sum += torch.sqrt(madnis_integrator.losses._variance(fi, qti,
                                     qsi) + madnis_integrator.losses.dtype_epsilon(f_true))
            abs_integral += torch.mean(fi.detach().abs() / qsi)
        return (stddev_sum / abs_integral) ** 2

    @staticmethod
    @madnis_integrator.losses.multi_channel_loss
    def kl_divergence_softclip(
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor,
        threshold: torch.Tensor = 30.0,
    ) -> torch.Tensor:
        """
        Computes the Kullback-Leibler divergence for two given sets of probabilities, ``f_true`` and
        ``q_test``. It uses importance sampling, i.e. the estimator is divided by an additional factor
        of ``q_sample``. A soft clipping function is applied to the sample weights.

        Args:
            f_true: normalized integrand values
            q_test: estimated function/probability
            q_sample: sampling probability
            channels: channel indices or None in the single-channel case
            threshold: approximate point of transition between linear and logarithmic behavior
        Returns:
            computed KL divergence
        """
        f_true = f_true.detach().abs()
        weight = f_true / q_sample
        weight /= weight.abs().mean()
        clipped_weight = MadnisIntegrator.softclip(weight, threshold)
        log_q = torch.log(q_test)
        log_f = torch.log(clipped_weight * q_sample +
                          madnis_integrator.losses.dtype_epsilon(f_true))
        return torch.mean(clipped_weight * (log_f - log_q))

    def export_state(self) -> 'MadnisIntegrator.MadnisState':
        flow_state = self.madnis.flow.state_dict()
        cwnet_state = self.madnis.cwnet.state_dict() if self.madnis.cwnet is not None else None
        optimizer_state = self.madnis.optimizer.state_dict() if self.madnis.optimizer is not None else None
        scheduler_state = self.madnis.scheduler.state_dict() if self.madnis.scheduler is not None else None
        return MadnisIntegrator.MadnisState(
            rng_state=torch.get_rng_state(),
            flow_state=flow_state,
            cwnet_state=cwnet_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
        )

    def import_state(self, state: 'MadnisIntegrator.MadnisState'):
        if not isinstance(state, MadnisIntegrator.MadnisState):
            raise ValueError("Invalid state type for MadnisIntegrator.")
        torch.set_rng_state(state.rng_state)
        self.madnis.flow.load_state_dict(state.flow_state)
        if state.cwnet_state is not None:
            if self.madnis.cwnet is None:
                print("WARNING: Cannot load CWNet state: Madnis integrator was not initialized with a CWNet.")
            else:
                self.madnis.cwnet.load_state_dict(state.cwnet_state)
        if state.optimizer_state is not None:
            if self.madnis.optimizer is None:
                print("WARNING: Cannot load optimizer state: Madnis integrator was not initialized with an optimizer.")
            else:
                self.madnis.optimizer.load_state_dict(state.optimizer_state)
        if state.scheduler_state is not None:
            self.madnis.scheduler = self._get_scheduler(T_max=1, scheduler_type=self.scheduler_type)
            self.madnis.scheduler.load_state_dict(state.scheduler_state)

    @dataclass
    class MadnisState:
        rng_state: Tensor
        flow_state: Dict[str, Any]
        cwnet_state: Dict[str, Any] | None
        optimizer_state: Dict[str, Any] | None
        scheduler_state: Dict[str, Any] | None
