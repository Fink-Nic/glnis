# type: ignore
import vegas
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, List, Callable, Literal
from numpy.typing import NDArray
from torch.types import Tensor
from symbolica import NumericalIntegrator, Sample, RandomNumberGenerator

import madnis.integrator as madnis_integrator
from glnis.core.accumulator import Accumulator, TrainingData, GraphProperties, LayerData
from glnis.core.integrand import MPIntegrand
from glnis.core.parser import SettingsParser


class Integrator(ABC):
    IDENTIFIER = "ABCIntegrator"

    def __init__(self,
                 integrand: MPIntegrand,
                 **uncaught_kwargs):
        self.integrand = integrand
        self.continuous_dim = integrand.continuous_dim
        self.discrete_dims = integrand.discrete_dims
        self.num_discrete_dims = len(integrand.discrete_dims)
        self.input_dim = self.continuous_dim + self.num_discrete_dims

    @abstractmethod
    def get_samples(self, n_points: int) -> LayerData:
        """Returns a LayerData object containing the samples to be fed into the integrand."""
        pass

    def integrate(self, n_points: int) -> Accumulator:
        layer_input = self.get_samples(n_points)

        return self.integrand.eval_integrand(layer_input)

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
            dtype=self.integrand.dtype,
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

        gl_result = Parser.get_gammaloop_integration_result()
        if gl_result is not None:
            integrand_kwargs['target_real'] = gl_result['result']['re']
            integrand_kwargs['target_imag'] = gl_result['result']['im']

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
                 seed=None,):
        super().__init__(
            integrand=integrand,)
        self.rng = np.random.default_rng(seed)

    def get_samples(self, n_points: int) -> LayerData:
        layer_input = self.init_layer_data(n_points)
        layer_input.continuous = np.random.random_sample(
            (n_points, self.continuous_dim))
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
                 **kwargs):
        super().__init__(
            integrand=integrand, **kwargs)
        self.vegas = vegas.Integrator(
            self.input_dim*[[0, 1],], **vegas_init_kwargs)

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
        total_wgt = disc_wgt * sampling_wgt * n_points
        layer_input.wgt *= total_wgt
        layer_input.update(self.IDENTIFIER)

        return layer_input

    @vegas.lbatchintegrand
    def _vegas_wrapper(self, x: NDArray) -> NDArray:
        layer_input = self.init_layer_data(len(x))
        layer_input.continuous = x[:, :self.continuous_dim]
        layer_input.discrete, layer_input.wgt = self._cont_to_discr(
            x[:, self.continuous_dim:])
        layer_input.update(self.IDENTIFIER)
        accumulated_result: TrainingData = self.integrand.eval_integrand(
            layer_input, 'training').modules[-1]

        return accumulated_result.training_result[0]


class HavanaIntegrator(Integrator):
    IDENTIFIER = "havana sampler"

    def __init__(self, integrand: MPIntegrand,
                 seed: int = 1337,
                 use_uniform: bool = False,
                 max_prob_ratio: float = 100,
                 n_continuous_bins: int = 128,
                 discrete_learning_rate: float = 1.5,
                 continuous_learning_rate: float = 1.5,
                 **kwargs):
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

        self.rng = RandomNumberGenerator(seed, 0)
        self.discrete_learning_rate = discrete_learning_rate
        self.continuous_learning_rate = continuous_learning_rate

    def train(self, nitn: int = 10, batch_size: int = 10000) -> str:
        report = [
            f"Training Havana for {nitn} iterations à {batch_size} samples:"]
        n_digits = len(str(nitn))
        for itn in range(nitn):
            samples, layer_input = self.get_samples(batch_size, True, True)
            accumulated_result = self.integrand.eval_integrand(
                layer_input, 'training')
            accumulated_result: TrainingData = accumulated_result.modules[-1]
            weighted_func_val = accumulated_result.training_result[0]
            self.havana.add_training_samples(
                samples, weighted_func_val.ravel().tolist())
            avg, err, chi_sq = self.havana.update(
                self.discrete_learning_rate, self.continuous_learning_rate)

            report.append(
                f"It {itn:0>{n_digits}}: {avg:.6e} +- {err:.6e}, chi={chi_sq:.3f}")

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
        for i in range(self.num_discrete_dims):
            discrete_wgt = np.take_along_axis(
                self.integrand.discrete_prior_prob_function(
                    discrete[:, :i], i),
                discrete[:, i].reshape(-1, 1))
            layer_input.wgt /= discrete_wgt  # * self.discrete_dims[i]
        if not training:
            weights = np.array([s.weights for s in samples])
            print(f"{weights.shape=}")
            weights = weights[:, -1]
            print(f"{weights.mean():.3e} +- {weights.std():.3e} (mean weight and stddev of current samples)")
            print(f"{layer_input.wgt.mean():.3e} +- {layer_input.wgt.std():.3e} (mean weight and stddev of layer input)")
            layer_input.wgt *= weights.reshape(-1, 1)
        layer_input.update(self.IDENTIFIER)

        if return_samples:
            return samples, layer_input

        return layer_input


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
                 ):
        super().__init__(integrand)
        import torch
        torch.set_default_dtype(torch.float64)

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
            discrete_dims_position="first",
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

    def old_integrate(self, n_points: int) -> madnis_integrator.IntegrationMetrics:
        return self.madnis.integration_metrics(n_points)

    def integrate(self, n_points: int) -> Accumulator:
        layer_input = self.get_samples(n_points)

        return self.integrand.eval_integrand(layer_input)

    def train(self, nitn: int = 10, batch_size: int = 10000):
        if self.use_scheduler:
            self.madnis.scheduler = self._get_scheduler(
                nitn, self.scheduler_type)
        self.madnis.train(nitn, self.callback, True)

    def get_samples(self, n_points: int) -> LayerData:
        layer_input = self.init_layer_data(n_points)

        with torch.no_grad():
            x_all, prob = self.madnis.flow.sample(
                n_points,
                return_prob=True,
                device=self.madnis.dummy.device,
                dtype=self.madnis.dummy.dtype,
            )
        layer_input.discrete = x_all[:,
                                     :self.num_discrete_dims].numpy(force=True)
        layer_input.continuous = x_all[:,
                                       self.num_discrete_dims:].numpy(force=True)
        layer_input.wgt /= prob.numpy(force=True).reshape(-1, 1)
        layer_input.update(self.IDENTIFIER)

        return layer_input

    def _madnis_eval(self, x_all: Tensor) -> Tensor:
        layer_input = self.init_layer_data(x_all.shape[0])
        layer_input.discrete = x_all[:,
                                     :self.num_discrete_dims].numpy(force=True)
        layer_input.continuous = x_all[:,
                                       self.num_discrete_dims:].numpy(force=True)
        layer_input.update(self.IDENTIFIER)

        accumulated_result = self.integrand.eval_integrand(
            layer_input, 'training')
        accumulated_result: TrainingData = accumulated_result.modules[-1]
        weighted_func_val = accumulated_result.training_result[0].flatten()
        torch_output = torch.from_numpy(
            weighted_func_val.astype(np.float64)).to(self.device)
        return torch_output

    def _madnis_discrete_prior_prob_function(self, indices: Tensor, dim: int = 0) -> Tensor:
        numpy_output = self.integrand.discrete_prior_prob_function(
            indices.numpy(force=True).astype(np.uint64), dim)
        torch_output = torch.from_numpy(
            numpy_output.astype(np.float64)).to(self.device)
        return torch_output

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
