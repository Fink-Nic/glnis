# type: ignore
import vegas
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, List, Callable, Literal
from numpy.typing import NDArray
from torch.types import Tensor

import madnis.integrator as madnis_integrator
from glnis.core.accumulator import Accumulator, TrainingData, GraphProperties, LayerData
from glnis.core.integrand import MPIntegrand
from glnis.core.parser import SettingsParser


class Integrator(ABC):
    IDENTIFIER = "ABCIntegrator"

    def __init__(self,
                 integrand: MPIntegrand,):
        self.integrand = integrand
        self.continuous_dim = integrand.continuous_dim
        self.discrete_dims = integrand.discrete_dims
        self.num_discrete_dims = len(integrand.discrete_dims)
        self.input_dim = self.continuous_dim + self.num_discrete_dims

    @abstractmethod
    def integrate(self, n_points: int) -> Accumulator:
        pass

    def train(self, nitn: int = 10, batch_size: int = 10000, ):
        print(f"Training not available for Integrator {self.IDENTIFIER}.")

    def _cont_to_discr(self, continuous: NDArray
                       ) -> Tuple[NDArray, NDArray]:
        n = len(continuous)
        if not continuous.shape[1] == len(self.discrete_dims):
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
            n_disc=len(self.discrete_dims),
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

        match integrator_type:
            case 'naive':
                return NaiveIntegrator(integrand)
            case 'vegas':
                return VegasIntegrator(integrand)
            case 'madnis':
                return MadnisIntegrator(integrand,
                                        integrator_kwargs=integrator_kwargs,)
            case _:
                return NaiveIntegrator(integrand)

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

    def integrate(self, n_points: int) -> Accumulator:
        layer_input = self.init_layer_data(n_points)
        layer_input.continuous = np.random.random_sample(
            (n_points, self.continuous_dim))
        discrete = self.rng.uniform(
            size=(n_points, len(self.discrete_dims)))
        layer_input.discrete, layer_input.wgt = self._cont_to_discr(discrete)
        layer_input.update(self.IDENTIFIER)

        return self.integrand.eval_integrand(layer_input)


class VegasIntegrator(Integrator):
    IDENTIFIER = 'vegas sampler'

    def __init__(self,
                 integrand: MPIntegrand,
                 **vegas_init_kwargs,):
        super().__init__(
            integrand=integrand,)
        self.integrator = vegas.Integrator(
            self.input_dim*[[0, 1],], **vegas_init_kwargs)

    def integrate(self, n_points: int) -> Accumulator:
        sampling_wgt, samples = self.integrator.sample(n_points, mode="lbatch")
        sampling_wgt = sampling_wgt.reshape(-1, 1)
        n_points = len(sampling_wgt)
        layer_input = self.init_layer_data(n_points)
        layer_input.continuous = samples[:, :self.continuous_dim]
        layer_input.discrete, disc_wgt = self._cont_to_discr(
            samples[:, self.continuous_dim:])
        # Because our accumulators take the average, not the sum like vegas wants us to
        total_wgt = disc_wgt * sampling_wgt * n_points
        layer_input.wgt *= total_wgt
        layer_input.update(self.IDENTIFIER)

        return self.integrand.eval_integrand(layer_input)

    def train(self, nitn: int = 10, batch_size: int = 10000, **vegas_kwargs) -> vegas._vegas.RAvg:
        return self.integrator(self._vegas_wrapper, nitn=nitn, neval=batch_size, **vegas_kwargs)

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


class MadnisIntegrator(Integrator):
    IDENTIFIER = 'madnis sampler'

    def __init__(self,
                 integrand: MPIntegrand,
                 integrator_kwargs: Dict[str, Any] = dict(
                     batch_size=1024,
                     learning_rate=1e-3,
                     discrete_model="transformer",
                     transformer=dict(
                         embedding_dim=64,
                         feedforward_dim=64,
                         heads=4,
                         mlp_units=64,
                         transformer_layers=1,),
                     use_scheduler=True,
                     n_train_for_scheduler=1000,
                     # loss_type: Literal["variance, variance_softclip, kl_divergence, kl_divergence_softclip"]
                     loss_type="kl_divergence",
                     # loss_kwargs: Dict[str, Any] = dict(),
                 ),
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

        if integrator_kwargs.pop('use_scheduler'):
            T_max = int(integrator_kwargs.pop('n_train_for_scheduler'))

            def scheduler(optimizer):
                return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        else:
            scheduler = None
        self.batch_size = integrator_kwargs['batch_size']

        match integrator_kwargs.pop('loss_type', 'kl_divergence').lower():
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
        discrete_flow_kwargs = integrator_kwargs.pop(
            integrator_kwargs['discrete_model'])
        integrator_kwargs.pop('transformer', 0)
        integrator_kwargs.pop('made', 0)

        self.madnis = madnis_integrator.Integrator(
            madnis_integrand,
            device=self.device,
            discrete_flow_kwargs=discrete_flow_kwargs,
            scheduler=scheduler,
            loss=loss,
            **integrator_kwargs,
        )
        self.callback = self._default_callback if callback is None else callback

    def integrate(self, n_points: int) -> madnis_integrator.IntegrationMetrics:
        return self.madnis.integration_metrics(n_points)

    def train(self, nitn: int = 10, batch_size: int = 10000):
        self.madnis.train(nitn, self.callback, True)

    def _madnis_eval(self, x_all: Tensor) -> Tensor:
        layer_input = self.init_layer_data(x_all.shape[0])
        layer_input.discrete = x_all[:,
                                     :self.num_discrete_dims].numpy(force=True)
        layer_input.continuous = x_all[:,
                                       self.num_discrete_dims:].numpy(force=True)
        layer_input.update(self.IDENTIFIER)

        output = self.integrand.eval_integrand(
            layer_input, 'training')
        output: TrainingData = output.modules[-1]
        weighted_func_val = output.training_result[0].flatten()
        torch_output = torch.from_numpy(
            weighted_func_val.astype(np.float64)).to(self.device)
        return torch_output

    def _madnis_discrete_prior_prob_function(self, indices: Tensor, dim: int = 0) -> Tensor:
        numpy_output = self.integrand.discrete_prior_prob_function(
            indices.numpy(force=True).astype(np.uint64), dim)
        torch_output = torch.from_numpy(
            numpy_output.astype(np.float64)).to(self.device)
        return torch_output

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
