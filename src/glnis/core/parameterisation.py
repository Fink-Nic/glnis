# type: ignore
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from numpy.typing import NDArray
from copy import deepcopy

import momtrop
from glnis.core.accumulator import GraphProperties, LayerData


type ParamOutput = Tuple[NDArray, NDArray, NDArray | None]


class Parameterisation(ABC):
    N_SPATIAL_DIMS = 3
    IDENTIFIER = "ABCParameterisation"

    def __init__(self,
                 graph_properties: GraphProperties,
                 next_param: 'Parameterisation' = None,
                 is_first_layer: bool = False,
                 **uncaught_kwargs):
        self.graph_properties = graph_properties
        self.next_param = next_param
        self.is_first_layer = is_first_layer

        self.layer_continuous_dim_in = self._layer_continuous_dim_in()
        self.layer_continuous_dim_out = self._layer_continuous_dim_out()
        self.layer_discrete_dims = self._layer_discrete_dims()
        self.layer_num_discrete_dims = len(self.layer_discrete_dims)

        self.chain_continuous_dim_in = self.get_chain_continuous_dim()
        self.chain_discrete_dims = self.get_chain_discrete_dims()

    @abstractmethod
    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray,
                            ) -> ParamOutput:
        """
        Args:
            continuous: continuous parameters
            discrete: discrete parameters
        Returns:
            jacobians, parameterised continuous output
        """
        pass

    def parameterise(self, layer_input: LayerData) -> LayerData:
        """
        Args:
            LayerData: Output of the previous Parameterisation layer (or sampler)
        Returns:
            LayerData: Parameterisation and data to be passed along the chain
        Raises:
            ValueError
        """
        param_input = self._to_layer_input(layer_input)
        param_output = self._layer_parameterise(*param_input)
        layer_output = self._to_layer_output(layer_input, *param_output)

        if self.next_param is None:
            return layer_output

        return self.next_param.parameterise(layer_output)

    def discrete_prior_prob_function(self, indices: NDArray, dim: int = 0) -> NDArray:
        """
        Args:
            indices: indices of the discrete channel
            dim: current index on dim=1 of generated indices
        Returns:
            torch tensor of shape (indices.shape[0], self.layer_num_discrete_dims):
            probability of the prior distribution for given indices.
            Default is flat probability distribution,
            zero if indices.shape[0] = self.layer_num_discrete_dims
        """
        if dim < self.layer_num_discrete_dims or self.next_param is None:
            return self._layer_prior_prob_function(indices)

        indices = indices[:, self.layer_num_discrete_dims:]
        dim -= self.layer_num_discrete_dims

        return self.next_param.discrete_prior_prob_function(indices, dim)

    def get_chain_discrete_dims(self) -> List[int]:
        """
        Returns:
            List of shape of the discrete dimensions of this and following layers in the
            parameterisation chain.
            Intended to be used to set the discrete dimensions for an Integrator.
        """
        if self.next_param is None:
            return self.layer_discrete_dims
        return self.layer_discrete_dims + self.next_param.get_chain_discrete_dims()

    def get_chain_continuous_dim(self) -> int:
        """
        Returns:
            The continuous dimension of this and following layers in the parameterisation chain.
            Intended to be used to set the number of continuous dimensions for an Integrator.
        """
        if self.next_param is None:
            return self.layer_continuous_dim_in

        layer_dim = self.layer_continuous_dim_in - self.layer_continuous_dim_out

        return layer_dim + self.next_param.get_chain_continuous_dim()

    def _to_layer_input(self, input: LayerData
                        ) -> Tuple[NDArray, NDArray]:
        """
        Returns the part of the input data that is relevant to the current layer, respecting
        the structure of the chain.

        Args:
            input: LayerData of the previous layer
        Returns:
            Continuous Samples: NDArray of shape(n_samples, self.continuous_dim)
            Discrete Samples: NDArray of shape(n_samples, <=len(self.discrete_dims))
        Raises:
            ValueError
        """
        continuous = input.continuous
        discrete = input.discrete
        if (n_dim := continuous.shape[1]) < self.chain_continuous_dim_in:
            raise ValueError(
                f"Layer {self.IDENTIFIER} has received {n_dim}-dimensional continuous input, "
                + f"expected at least {self.chain_continuous_dim_in}.")
        n_disc = len(self.chain_discrete_dims)
        if (n_dim := discrete.shape[1]) < n_disc:
            raise ValueError(
                f"Layer {self.IDENTIFIER} has received {n_dim}-dimensional discrete input, "
                + f"expected at least {n_disc}.")

        continuous = continuous[:, :self.layer_continuous_dim_in]

        if discrete.shape[1] > self.layer_num_discrete_dims:
            discrete = discrete[:, :self.layer_num_discrete_dims]

        return continuous, discrete

    def _to_layer_output(self, layer_input: LayerData,
                         jac_param: NDArray,
                         cont_param: NDArray,
                         disc_param: NDArray | None,) -> LayerData:
        """
        Returns the output of the current layer in a form that respects the structure of the chain,
        in order to be passed down the chain.

        Args:
            layer_input: LayerData of the previous layer
            *param_output: Output generated by the parameterisation step of the current layer
        Returns:
            LayerData
        """
        if disc_param is None:
            disc_param = np.zeros(
                (layer_input.n_points, 0), dtype=layer_input.dtype)
        # Pass along potential additional input that is required down the chain
        n_cont = layer_input._active_structure[layer_input.POSITIONS['continuous']]
        n_disc = layer_input._active_structure[layer_input.POSITIONS['discrete']]

        if n_cont > self.layer_continuous_dim_in:
            cont_pass = layer_input.continuous[:,
                                               self.layer_continuous_dim_in:]
            cont_param = np.hstack([cont_param, cont_pass])
        if n_disc > self.layer_num_discrete_dims:
            disc_pass = layer_input.discrete[:, self.layer_num_discrete_dims:]
            disc_param = np.hstack([disc_param, disc_pass])

        # Update the data
        layer_input.jac *= jac_param
        if self.next_param is None:
            layer_input.momenta = cont_param
        else:
            layer_input.continuous = cont_param
        layer_input.discrete = disc_param
        layer_input.update(self.IDENTIFIER)

        return layer_input

    def _layer_prior_prob_function(self, indices: NDArray) -> NDArray:
        num_disc_input = indices.shape[1]
        if num_disc_input == self.layer_num_discrete_dims:
            return np.zeros_like(indices, dtype=np.float64)

        disc_dim = self.layer_discrete_dims[num_disc_input]
        norm_factor = disc_dim - indices.shape[1]
        prior = np.ones(
            (len(indices), disc_dim), dtype=np.float64)
        if num_disc_input == 0:
            return prior / norm_factor

        rows = np.repeat(np.arange(len(indices)), num_disc_input)
        prior[rows, indices.flatten()] = 0

        return prior / norm_factor

    def _layer_discrete_dims(self) -> List[int]:
        """
        Returns:
            List of shape of the discrete dimensions of this layer in the parameterisation chain.
            Intended to be used to initialize the value for self.discrete_dims
        """
        return []

    def _layer_continuous_dim_in(self) -> int:
        """
        Intended to be used to initialize the value for self.layer_continuous_dim_in.

        Returns:
            The continuous dimension of the input of this layer in the parameterisation chain.
        """
        return self.N_SPATIAL_DIMS*self.graph_properties.n_loops

    def _layer_continuous_dim_out(self) -> int:
        """
        Intended to be used to initialize the value for self.layer_continuous_dim_out.

        Returns:
            The continuous dimension of the output of this layer in the parameterisation chain.
        """
        return self.N_SPATIAL_DIMS*self.graph_properties.n_loops

    def set_params(self, **kwargs: Dict):
        """Update the parameters of the parameterisation."""
        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")


class LayeredParameterisation:
    IDENTIFIER = "layered parameterisation"

    def __init__(self, graph_properties: GraphProperties,
                 param_settings: List[Dict[str, Any]],):
        param_layers: List[Parameterisation] = []
        num_layers = len(param_settings)
        for i_layer, kdict in enumerate(param_settings):
            kwargs = deepcopy(kdict)
            if not "parameterisation_type" in kwargs.keys():
                raise KeyError(
                    "Each parameterisation layer must specify its parameterisation type.")

            param_type: str = kwargs.pop("parameterisation_type")
            is_first_layer = (i_layer + 1) == num_layers
            next_param = None if i_layer == 0 else param_layers[-1]
            kwargs.update(dict(is_first_layer=is_first_layer,
                               next_param=next_param,
                               graph_properties=graph_properties,))

            match param_type.lower():
                case "momtrop":
                    p = MomtropParameterisation(**kwargs)
                case "spherical":
                    p = SphericalParameterisation(**kwargs)
                case "inv_spherical":
                    p = InverseSphericalParameterisation(**kwargs)
                case "kaapo":
                    p = KaapoParameterisation(**kwargs)
                case "rkaapo":
                    p = RKaapoParameterisation(**kwargs)
                case "mc_layer":
                    if next_param is None:
                        raise ValueError(
                            "MC layer must be passed after a parameterisation.")
                    kwargs.update(dict(is_first_layer=next_param.is_first_layer,
                                       next_param=next_param.next_param,
                                       param=next_param,))
                    match kwargs.pop('subtype', 'ose').lower():
                        case "ose":
                            p = OSEMCLayer(**kwargs)
                        case "fermi":
                            p = FermiMCLayer(**kwargs)
                case _:
                    raise NotImplementedError(
                        f"Parameterisation {param_type} has not been implemented.")
            param_layers.append(p)

        self.param = param_layers[-1]
        self.continuous_dim = self.param.chain_continuous_dim_in
        self.discrete_dims = self.param.chain_discrete_dims

    def parameterise(self, layer_input: LayerData) -> LayerData:
        return self.param.parameterise(layer_input)

    def discrete_prior_prob_function(self, indices: NDArray, dim: int = 0) -> NDArray:
        return self.param.discrete_prior_prob_function(indices, dim)


class MomtropParameterisation(Parameterisation):
    IDENTIFIER = "momtrop param"

    def __init__(self, overwrite_edge_weight: float | List[float] | bool = False,
                 sample_discrete: bool = True,
                 **kwargs: Dict[str, Any]):
        self.sample_discrete = sample_discrete
        self.graph_properties: GraphProperties = kwargs["graph_properties"]
        match overwrite_edge_weight:
            case bool():
                pass
            case int() | float():
                self.graph_properties.momtrop_edge_weight = self.graph_properties.n_edges * \
                    [float(overwrite_edge_weight)]
            case [_, *_]:
                if not len(overwrite_edge_weight) == self.graph_properties.n_edges:
                    raise ValueError("If provided as a sequence, the number of momtrop "
                                     + "edgeweights must match the number of propagators.")
                self.graph_properties.momtrop_edge_weight = overwrite_edge_weight
            case _:
                pass

        mt_edges = [
            momtrop.Edge(tuple(src_dst), ismassive, weight) for src_dst, ismassive, weight
            in zip(self.graph_properties.edge_src_dst_vertices,
                   self.graph_properties.edge_ismassive,
                   self.graph_properties.momtrop_edge_weight)
        ]
        assym_graph = momtrop.Graph(
            mt_edges, self.graph_properties.graph_external_vertices)
        momentum_shifts = [momtrop.Vector(*shift) for shift
                           in self.graph_properties.edge_momentum_shifts]
        self.momtrop_edge_data = momtrop.EdgeData(
            self.graph_properties.edge_masses, momentum_shifts)
        self.momtrop_sampler = momtrop.Sampler(
            assym_graph, self.graph_properties.graph_signature)
        self.momtrop_sampler_settings = momtrop.Settings(False, False)
        super().__init__(**kwargs)

    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray,
                            ) -> ParamOutput:
        dtype = continuous.dtype
        if discrete.size == 0:
            samples = self.momtrop_sampler.sample_batch(
                continuous.tolist(), self.momtrop_edge_data, self.momtrop_sampler_settings)
        else:
            samples = self.momtrop_sampler.sample_batch(
                continuous.tolist(), self.momtrop_edge_data, self.momtrop_sampler_settings,
                self._get_graph_from_edges_removed(discrete))

        jac = np.array(samples.jacobians, dtype=dtype).reshape(-1, 1)
        momentum = np.array(
            samples.loop_momenta, dtype=dtype).reshape(len(continuous), -1)

        return jac, momentum, None

    def _get_graph_from_edges_removed(self, edges_removed: NDArray | None = None
                                      ) -> List[List[int]]:
        """
        Args:
            edges_removed: List of the edge indices that have already been forced
        Returns:
            List of shape (n_edges,) that appends the as-yet unforced edges to edges_removed
        """
        full_graph = np.arange(self.layer_num_discrete_dims)
        if edges_removed is None:
            return [full_graph.tolist()]

        full_graph = np.tile(full_graph, (len(edges_removed), 1))
        if edges_removed.shape[1] == 0:
            return full_graph.tolist()

        removed_mask = np.zeros(
            (edges_removed.shape[0], self.layer_num_discrete_dims), dtype=np.bool)
        rows = np.repeat(
            np.arange(edges_removed.shape[0]), edges_removed.shape[1])
        removed_mask[rows, edges_removed.flatten()] = 1
        # Append the edges that are not in discrete, meaning where onehot is zero
        remaining_edges = full_graph[~removed_mask].reshape(
            len(edges_removed), -1)

        return np.hstack([edges_removed, remaining_edges]).astype(np.uint64).tolist()

    def _layer_prior_prob_function(self, indices: NDArray) -> NDArray:
        rust_result = self.momtrop_sampler.predict_discrete_probs(
            indices.tolist())

        return np.array(rust_result)

    def _layer_continuous_dim_in(self) -> int:
        return self.momtrop_sampler.get_dimension()

    def _layer_discrete_dims(self) -> List[int]:
        if not self.sample_discrete:
            return []
        n_edges = self.graph_properties.n_edges
        return n_edges * [n_edges]


class SphericalParameterisation(Parameterisation):
    IDENTIFIER = "spherical param"

    def __init__(self,
                 conformal_scale: float = 1.,
                 origins: NDArray | List[List[float]
                                         ] | List[float] | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.conformal_scale = conformal_scale
        self.n_loops = self.graph_properties.n_loops
        if origins is None or origins == 0.:
            self.origins = self.n_loops * [None]
        else:
            self.origins = np.array(origins)
            if self.origins.ndim == 1:
                self.origins = np.tile(self.origins, (self.n_loops, 1))

    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray,
                            ) -> ParamOutput:
        momentum = np.zeros_like(continuous)

        # Constant part of the jacobian
        jac = np.ones((len(continuous), 1), dtype=continuous.dtype)
        jac *= (4*np.pi * self.conformal_scale**3)**self.n_loops

        for i_loop in range(self.n_loops):
            # The MC layer handles momentum shifts
            if discrete.size > 0:
                origin = None
            else:
                origin = self.origins[i_loop]
            if discrete.size == 0:
                discrete = np.zeros((continuous.shape[0], 1), dtype=np.uint64)
            _start = self.N_SPATIAL_DIMS*i_loop
            _end = self.N_SPATIAL_DIMS*(i_loop + 1)

            xs = continuous[:, _start: _end]
            x, y, z = np.hsplit(xs, [1, 2])

            r = x/(1-x)
            cos_az = (2*y-1)
            sin_az = np.sqrt(1 - cos_az**2)
            pol = 2*np.pi*z

            _start = self.N_SPATIAL_DIMS*i_loop
            _end = self.N_SPATIAL_DIMS*(i_loop + 1)
            ks = self.conformal_scale*r * \
                np.hstack(
                    [sin_az * np.cos(pol), sin_az * np.sin(pol), cos_az])
            if origin is not None:
                ks -= origin
            momentum[:, _start: _end] = ks
            # Calculate the jacobian determinant
            jac *= x**2 / (1 - x)**4

        # Transform the loop momenta back to the LMB of the graph
        inv_transform = self.graph_properties.channel_inv_transforms[discrete.flatten(
        )]
        momentum = inv_transform @ momentum.reshape(-1, self.n_loops, 3)

        return jac, momentum, None


class InverseSphericalParameterisation(Parameterisation):
    IDENTIFIER = "inverse spherical param"

    def __init__(self,
                 conformal_scale: float,
                 origins: NDArray | List[List[float]
                                         ] | List[float] | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.conformal_scale = conformal_scale
        self.n_loops = self.graph_properties.n_loops
        if origins is None or origins == 0.:
            self.origins = self.n_loops * [None]
        else:
            self.origins = np.array(origins)
            if self.origins.ndim == 1:
                self.origins = np.tile(self.origins, (self.n_loops, 1))

    def _layer_parameterise(self, continuous: NDArray, _: NDArray,
                            ) -> Tuple[NDArray, NDArray]:
        xs = np.zeros_like(continuous)

        # Constant part of the jacobian
        jac = np.ones((len(continuous), 1), dtype=continuous.dtype)
        jac /= (4*np.pi * self.conformal_scale**3)**self.n_loops

        for i_loop in range(self.n_loops):
            origin = self.origins[i_loop]
            _start = self.N_SPATIAL_DIMS*i_loop
            _end = self.N_SPATIAL_DIMS*(i_loop + 1)
            ks = continuous[:, _start: _end]
            if origin is not None:
                ks += origin

            k0, k1, k2 = np.hsplit(ks, [1, 2])

            r = np.linalg.norm(ks, axis=1).reshape(-1, 1)
            cos_az = k2 / r
            tan_pol = (k1 / k0).reshape(-1, 1)
            pol: NDArray = np.arctan(tan_pol)
            # Accounting for missing quadrants of arctan
            pol += np.pi*(1 - np.sign(k0))/2 * np.sign(k1)
            pol += np.pi*(1 - np.sign(pol))

            r /= self.conformal_scale
            x = r / (1 + r)
            y = (cos_az + 1) / 2
            z = pol / 2 / np.pi
            xs[:, _start: _end] = np.hstack([x, y, z])

            # Calculate the jacobian determinant
            jac /= x**2 / (1 - x)**4

        return jac, xs, None


class KaapoParameterisation(Parameterisation):
    IDENTIFIER = "kaapo param"

    def __init__(self, mu: List[float] | float = np.pi,
                 a: float = 0.5,
                 b: float = 1.0,
                 vary_a: bool = False,
                 a_min: float = 0.2,
                 **kwargs):
        self.mu = mu
        if not type(self.mu) == list:
            self.mu: list[float] = self.graph_properties.n_edges*[self.mu]

        self.a = a
        self.b = b
        self.vary_a = vary_a
        self.a_min = a_min
        super().__init__(**kwargs)

    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray
                            ) -> ParamOutput:
        if discrete.size == 0:
            discrete = np.zeros((continuous.shape[0], 1), dtype=np.uint64)
        # For easier reading
        n_loops = self.graph_properties.n_loops
        n_points = continuous.shape[0]
        dtype = continuous.dtype
        if self.vary_a:
            a = self.a_min + (1. - self.a_min)*continuous[:, -1].reshape(-1, 1)
        else:
            a = self.a
        b = self.b

        momentum = np.zeros((n_points, 3*n_loops), dtype=dtype)

        # The constant part of the jacobian
        jac = np.ones((n_points, 1), dtype=dtype)
        jac *= (4 * np.pi / a / b**a)**n_loops

        for i_loop in range(n_loops):
            basis_edge = self.graph_properties.lmb_array[discrete, i_loop]
            m_e = np.array(self.graph_properties.edge_masses)[basis_edge]
            mu = np.array(self.mu)[basis_edge]
            p_F = np.clip(
                mu**2 - m_e**2, a_min=0., a_max=None)**0.5

            _start = self.N_SPATIAL_DIMS*i_loop
            _end = self.N_SPATIAL_DIMS*(i_loop + 1)

            xs = continuous[:, _start:_end]
            x1, x2, x3 = np.hsplit(xs, [1, 2])

            cos_az = (2*x2-1)
            sin_az = np.sqrt(1 - cos_az**2)
            pol = 2*np.pi*x3

            # Discriminator around the fermi surface and origin
            peak_F: NDArray = b**a * x1 / (1 - x1) - p_F**a

            # Radial component
            h_c = p_F + np.sign(peak_F) * np.abs(peak_F)**(1 / a)

            # Standard spherical parameterisation, scaled by h_c
            k_vec = h_c * np.hstack(
                [sin_az * np.cos(pol), sin_az * np.sin(pol), cos_az])
            momentum[:, _start: _end] = k_vec

            # Calculate the jacobian
            jac *= h_c**2 * np.abs(peak_F)**(1 / a - 1)
            jac *= (np.sign(peak_F)*np.abs(peak_F) + p_F**a + b**a)**2

        # Transform the loop momenta back to the LMB of the graph
        inv_transform = self.graph_properties.channel_inv_transforms[discrete.flatten(
        )]
        momentum = inv_transform @ momentum.reshape(-1, n_loops, 3)

        return jac, momentum.reshape(n_points, -1), None

    def _layer_continuous_dim_in(self) -> int:
        c_dim = self.N_SPATIAL_DIMS*self.graph_properties.n_loops
        if self.vary_a:
            c_dim += 1
        return c_dim


class RKaapoParameterisation(Parameterisation):
    IDENTIFIER = "reduced kaapo param"

    def __init__(self, mu: List[float] | float = np.pi,
                 a: float = 0.5,
                 b: float = 1.0,
                 vary_a: bool = False,
                 a_min: float = 0.2,
                 **kwargs):
        self.mu = mu
        if not type(self.mu) == list:
            self.mu: list[float] = self.graph_properties.n_edges*[self.mu]

        self.a = a
        self.b = b
        self.vary_a = vary_a
        self.a_min = a_min
        super().__init__(**kwargs)

    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray
                            ) -> ParamOutput:
        if discrete.size == 0:
            discrete = np.zeros((continuous.shape[0], 1), dtype=np.uint64)
        # For easier reading
        n_loops = self.graph_properties.n_loops
        n_points = continuous.shape[0]
        dtype = continuous.dtype
        if self.vary_a:
            a = self.a_min + (1. - self.a_min)*continuous[:, -1].reshape(-1, 1)
        else:
            a = self.a
        b = self.b

        momentum = np.zeros((n_points, 3*n_loops), dtype=dtype)

        # The constant part of the jacobian
        jac = np.ones((n_points, 1), dtype=dtype)
        jac *= (4 * np.pi / a / b**a)**n_loops

        for i_loop in range(n_loops):
            basis_edge = self.graph_properties.lmb_array[discrete, i_loop]
            m_e = np.array(self.graph_properties.edge_masses)[basis_edge]
            mu = np.array(self.mu)[basis_edge]
            p_F = np.clip(
                mu**2 - m_e**2, a_min=0., a_max=None)**0.5

            _start = self.N_SPATIAL_DIMS*i_loop
            _end = self.N_SPATIAL_DIMS*(i_loop + 1)

            if i_loop == 0:
                x1 = continuous[:, 0].reshape(-1, 1)
                cos_az = np.ones(
                    (continuous.shape[0], 1), dtype=continuous.dtype)
                sin_az = np.zeros(
                    (continuous.shape[0], 1), dtype=continuous.dtype)
                pol = 0
            elif i_loop == 1:
                x1 = continuous[:, 1].reshape(-1, 1)
                cos_az = (2*continuous[:, 2] - 1).reshape(-1, 1)
                sin_az = np.sqrt(1 - cos_az**2)
                pol = 0
            else:
                xs = continuous[:, _start -
                                self.N_SPATIAL_DIMS: _end-self.N_SPATIAL_DIMS]
                x1, x2, x3 = np.hsplit(xs, [1, 2])

                cos_az = (2*x2-1)
                sin_az = np.sqrt(1 - cos_az**2)
                pol = 2*np.pi*x3

            # Discriminator around the fermi surface and origin
            peak_F: NDArray = b**a * x1 / (1 - x1) - p_F**a

            # Radial component
            h_c = p_F + np.sign(peak_F) * np.abs(peak_F)**(1 / a)

            # Standard spherical parameterisation, scaled by h_c
            k_vec = h_c * np.hstack(
                [sin_az * np.cos(pol), sin_az * np.sin(pol), cos_az])
            momentum[:, _start: _end] = k_vec

            # Calculate the jacobian
            jac *= h_c**2 * np.abs(peak_F)**(1 / a - 1)
            jac *= (np.sign(peak_F)*np.abs(peak_F) + p_F**a + b**a)**2

        # Transform the loop momenta back to the LMB of the graph
        inv_transform = self.graph_properties.channel_inv_transforms[discrete.flatten(
        )]
        momentum = inv_transform @ momentum.reshape(-1, n_loops, 3)

        return jac, momentum.reshape(n_points, -1), None

    def _layer_continuous_dim_in(self) -> int:
        if self.graph_properties.n_loops == 1:
            c_dim = 1
        else:
            c_dim = self.N_SPATIAL_DIMS*(self.graph_properties.n_loops - 1)
        if self.vary_a:
            c_dim += 1
        return c_dim


class MCLayer(Parameterisation, ABC):
    IDENTIFIER = "MC layer"

    def __init__(self,
                 param: Parameterisation,
                 **kwargs):
        self.param = param
        self.IDENTIFIER += f" : {self.param.IDENTIFIER}"
        super().__init__(**kwargs)

        self.lmbs = self.graph_properties.lmb_array
        self.n_channels = self.graph_properties.n_channels
        self.n_loops = self.graph_properties.n_loops

        self.shifts = np.array(self.graph_properties.edge_momentum_shifts)
        self.channel_shifts = self.shifts[self.lmbs]
        self.channel_masses = np.array(
            self.graph_properties.edge_masses)[self.lmbs]

    def _layer_parameterise(self, continuous: NDArray, discrete: NDArray) -> ParamOutput:
        jac, momentum, _ = self.param._layer_parameterise(
            continuous, discrete)
        # Get the edge indices of the channel LMBs
        edges = self.lmbs[discrete]
        # Perform the inverse momentum shift
        sample_shifts = self.shifts[edges].reshape(-1, 3*self.n_loops)
        momentum -= sample_shifts
        # The MC weight factor is calculated in EMR (edge momentum representation)
        jac *= self._mc_weight(momentum, discrete).reshape(-1, 1)

        return jac, momentum, None

    @abstractmethod
    def _mc_weight(self, continuous: NDArray, discrete: NDArray) -> NDArray:
        return np.ones((continuous.shape[0], 1), dtype=continuous.dtype)

    def _layer_discrete_dims(self) -> List[int]:
        return [self.graph_properties.n_channels]

    def _layer_continuous_dim_in(self) -> int:
        return self.param._layer_continuous_dim_in()


class OSEMCLayer(MCLayer):
    IDENTIFIER = "OSE MC Layer"

    def __init__(self,
                 ose_exponent: float = 2.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.ose_exponent = ose_exponent

    def _mc_weight(self, continuous: NDArray, discrete: NDArray) -> NDArray:
        momentum = continuous.reshape(-1, self.n_loops, 3)
        # Need to calculate the e-surface term for all lmbs
        # Transform to the edge momentum basis
        # shape: (n_samples, n_channels, n_loops)
        edge_momentum_squared = np.sum(
            (self.graph_properties.channel_transforms @ momentum.reshape(
                -1, 1, self.n_loops, 3) + self.channel_shifts)**2, axis=3)
        # shape: (n_samples, n_channels, n_loops) -- prod --> (n_samples, n_channels)
        all_e_surface_terms = np.prod(
            edge_momentum_squared + self.channel_masses**2, axis=2)**(-self.ose_exponent/2.)
        mc_weight = np.take_along_axis(
            all_e_surface_terms, discrete, axis=1)
        # Normalize and return
        return mc_weight / np.sum(all_e_surface_terms, axis=1).reshape(-1, 1)


class FermiMCLayer(MCLayer):
    IDENTIFIER = "Fermi MC Layer"

    def __init__(self,
                 ose_exponent: float = 2.0,
                 fermi_exponent: float = 0.5,
                 set_bosonic_edge_to_one: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.ose_exponent = ose_exponent
        self.fermi_exponent = fermi_exponent
        self.set_bosonic_edge_to_one = set_bosonic_edge_to_one
        if hasattr(self.param, 'mu'):
            self.channel_mu = np.array(self.param.mu)[self.lmbs]
        else:
            self.channel_mu = np.zeros((self.n_channels, self.n_loops))

    def _mc_weight(self, continuous: NDArray, discrete: NDArray) -> NDArray:
        momentum = continuous.reshape(-1, self.n_loops, 3)
        self.param: KaapoParameterisation
        # Need to calculate the fermi surface term for all lmbs
        # Transform to the edge momentum basis
        # shape: (n_samples, n_channels, n_loops)
        edge_momentum_squared = np.sum(
            (self.graph_properties.channel_transforms @ momentum.reshape(
                -1, 1, self.n_loops, 3) + self.channel_shifts)**2, axis=3)
        # shape: (n_samples, n_channels, n_loops) -- prod --> (n_samples, n_channels)
        all_e_surface_terms = np.sqrt(
            edge_momentum_squared + self.channel_masses**2)
        # shape: (n_samples, n_channels, n_loops) -- prod --> (n_samples, n_channels)
        all_fermi_surface_terms = all_e_surface_terms - self.channel_mu
        if self.set_bosonic_edge_to_one:
            bosonic_edge_mask = self.channel_mu == 0.
            all_fermi_surface_terms[:, bosonic_edge_mask] = 1.
        all_fermi_surface_terms = np.abs(np.prod(
            all_fermi_surface_terms, axis=2))**(-self.fermi_exponent)
        # Re-assigning to save at least a smidgen of memory
        all_e_surface_terms = np.prod(
            all_e_surface_terms, axis=2)**(-self.ose_exponent)
        all_fermi_surface_terms *= all_e_surface_terms
        mc_weight = np.take_along_axis(
            all_fermi_surface_terms, discrete, axis=1)
        # Normalize and return
        return mc_weight / np.sum(all_fermi_surface_terms, axis=1).reshape(-1, 1)
