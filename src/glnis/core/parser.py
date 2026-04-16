# type: ignore
from symbolica import E, S, Expression
from typing import Dict, Tuple, List, Any, Set
from copy import deepcopy
from pathlib import Path
import momtrop
import pydot
import json
import tomllib
from glnis.utils.helpers import overwrite_settings, verify_path, shell_print
from glnis.core.accumulator import GraphProperties, IntegrationResult


class ModelParser:
    def __init__(self, model_path: str | Path, from_string=False):
        if from_string:
            self.model_path = "Loaded from existing model string."
            self.model = json.loads(model_path)
        else:
            self.model_path = verify_path(model_path, suffix=".json")
            with self.model_path.open("r") as f:
                self.model = json.load(f)

    def get_particle_from_identifier(self, identifier_name: str, value) -> Dict:
        particle_match = None
        for particle in self.model['particles']:
            try:
                if particle[identifier_name] == value:
                    particle_match = particle
            except:
                pass

        if particle_match is None:
            raise KeyError(
                f"Particle with {identifier_name}='{value}' does not exist in model '{self.model_path}'.")

        return particle_match

    def get_particle_parameter_from_identifier(self,
                                               identifier_name: str,
                                               identifier_value,
                                               parameter_name: str):
        particle_match = self.get_particle_from_identifier(
            identifier_name, identifier_value)
        try:
            model_parameter_name = particle_match[parameter_name]
        except KeyError:
            raise KeyError(
                f"Particle with '{identifier_name}' = '{identifier_value}' "
                + f"does not have parameter '{parameter_name}'.")

        if model_parameter_name == 'ZERO':
            return [0., 0.]

        parameter_match = None
        for parameter in self.model['parameters']:
            try:
                if parameter['name'] == model_parameter_name:
                    parameter_match = parameter['value']
            except:
                pass

        if parameter_match is None:
            raise KeyError(f"The model '{self.model_path}' does not specify a value for "
                           + f"the parameter '{model_parameter_name}'.")

        return parameter_match

    def get_particle_parameter_from_name(self, particle_name: str, parameter_name: str):
        return self.get_particle_parameter_from_identifier('name', particle_name, parameter_name)

    def get_particle_mass_from_name(self, particle_name: str):
        return self.get_particle_parameter_from_identifier('name', particle_name, 'mass')


class DotParser:
    def __init__(self, dot_file: str | Path, model: str | Path | ModelParser,
                 dot_from_string: bool = False,):
        if dot_from_string:
            self.graph_file = pydot.graph_from_dot_data(dot_file)
        else:
            dot_file = verify_path(dot_file, suffix=".dot")
            self.graph_file = pydot.graph_from_dot_file(str(dot_file))
        if type(model) == ModelParser:
            self.Model = model
        else:
            self.Model = ModelParser(model)

    def get_dot_graph(self, graph_id: int):
        return self.graph_file[graph_id]

    def infer_dependent_momentum(self,
                                 ext_momenta: List[List[float]],
                                 ext_sigs: List[int],
                                 dependent_momentum_index: int) -> List[List[float]]:
        # Infering the dependent momentum from momentum conservation
        if len(ext_momenta) == 0:
            return []
        if not len(ext_momenta) == len(ext_sigs):
            raise ValueError("Length of external momenta and external signatures must match.")
        dmi = dependent_momentum_index
        dm_sig = ext_sigs[dmi]
        dependent_momentum = 4*[0.]
        exclusive_external_sigs = ext_sigs[:dmi] + ext_sigs[dmi+1:]
        for momentum, sig in zip(ext_momenta, exclusive_external_sigs):
            dependent_momentum[0] -= dm_sig*sig*momentum[0]
            dependent_momentum[1] -= dm_sig*sig*momentum[1]
            dependent_momentum[2] -= dm_sig*sig*momentum[2]
            dependent_momentum[3] -= dm_sig*sig*momentum[3]
        ext_momenta = ext_momenta[:dmi] + \
            [dependent_momentum] + ext_momenta[dmi:]

        return ext_momenta

    def get_external_signature(self, graph_id: int = 0) -> List[int]:
        graph = self.get_dot_graph(graph_id)
        edges = graph.get_edges()

        ext_sigs = len(edges)*[None]
        for edge in edges:
            src_split = edge.get_source().split(':')
            dst_split = edge.get_destination().split(':')
            if edge.get('source') is None:
                ext_sigs[int(dst_split[1])] = 1
            elif edge.get('sink') is None:
                ext_sigs[int(src_split[1])] = -1
        ext_sigs = [sig for sig in ext_sigs if sig is not None]

        return ext_sigs

    def get_graph_properties(self, graph_id: int,
                             ext_momenta: List[List[float]],) -> GraphProperties:

        # External momenta
        n_ext_mom = len(ext_momenta)

        # Dot graph
        graph = self.get_dot_graph(graph_id)
        edges: List[pydot.Edge] = graph.get_edges()
        vertices = graph.get_nodes()

        VERTICES: list[pydot.Node] = []
        LMB_EDGES: list[pydot.Edge] = []
        EXT_VERTICES: Set[pydot.Node] = set()
        INT_EDGES: list[pydot.Edge] = []

        # Filter out the external vertices
        for vert in vertices:
            if vert.get('num') is not None:
                VERTICES.append(vert)

        # Add vertex ID for momtrop
        for v_id, vert in enumerate(VERTICES):
            vert.set('v_id', v_id)

        # Filter edges and add additional attributes for momtrop
        EXT_SIGNATURES = []
        for edge in edges:
            src_split = edge.get_source().split(':')
            dst_split = edge.get_destination().split(':')
            edge.set('src', src_split[0])
            edge.set('dst', dst_split[0])

            if edge.get('lmb_id') is not None:
                LMB_EDGES.append(edge)

            if edge.get('source') is None:
                # Incoming external momentum
                EXT_VERTICES.add(graph.get_node(edge.get("dst"))[0])
                EXT_SIGNATURES.append(1)
            elif edge.get('sink') is None:
                # Outgoing external momentum
                EXT_VERTICES.add(graph.get_node(edge.get("src"))[0])
                EXT_SIGNATURES.append(-1)
            else:
                INT_EDGES.append(edge)
                particle_name = edge.get('particle')[1:-1]
                edge.set('mass', self.Model.get_particle_mass_from_name(
                    particle_name)[0])
                src_vert = graph.get_node(edge.get('src'))[0]
                dst_vert = graph.get_node(edge.get('dst'))[0]
                edge.set('src_id', src_vert.get('v_id'))
                edge.set('dst_id', dst_vert.get('v_id'))

        # Reconstruct the dependent external momentum from momentum conservation
        if 'dependent' in ext_momenta:
            dmi = ext_momenta.index('dependent')
            ext_momenta = self.infer_dependent_momentum(
                ext_momenta, EXT_SIGNATURES, dependent_momentum_index=dmi)

        # Symbolica setup for LMB representation parsing
        # P: External momenta
        # K: Internal momenta
        # x_, a_: wildcards
        P, K = S('P', 'K')
        x_, a_ = S('x_', 'a_')

        # Set up momtrop sampler
        n_loops = len(LMB_EDGES)

        graph_externals = sorted([v.get("v_id") for v in EXT_VERTICES])
        graph_signature = []
        edge_momentum_shifts = []
        edge_src_dst_vertices = []
        edge_masses = []
        edge_external_sigs = []

        for edge in INT_EDGES:
            # Generate the momtrop edge
            src_id = edge.get('src_id')
            dst_id = edge.get('dst_id')
            mass = edge.get('mass')
            edge_src_dst_vertices.append([src_id, dst_id])
            edge_masses.append(mass)

            # LMB representation parsing
            e: Expression = E(edge.get('lmb_rep')[1:-1])
            e = e.replace(P(x_, a_), P(x_))
            e = e.replace(K(x_, a_), K(x_))
            lmb_sig = [int(e.coefficient(K(lmb_id)).to_sympy())
                       for lmb_id in range(n_loops)]
            graph_signature.append(lmb_sig)

            edge_external_sig = [float(e.coefficient(P(ext_id)).to_sympy())
                                 for ext_id in range(n_ext_mom)]
            momentum_shift = [0. for _ in range(3)]
            for coeff, ext_mom in zip(edge_external_sig, ext_momenta):
                for i in range(3):
                    momentum_shift[i] += coeff*ext_mom[i+1]

            edge_momentum_shifts.append(momentum_shift)
            edge_external_sigs.append(edge_external_sig)

        return GraphProperties(
            edge_src_dst_vertices=edge_src_dst_vertices,
            edge_masses=edge_masses,
            edge_momentum_shifts=edge_momentum_shifts,
            graph_external_vertices=graph_externals,
            graph_signature=graph_signature,
            edge_external_sigs=edge_external_sigs,
            external_momenta=ext_momenta,
        )


class SettingsParser:
    def __init__(self, settings: str | Path | Dict = "settings/default.toml",
                 _from_existing: bool = False):
        if not isinstance(settings, dict):
            settings_path = Path(settings)
            settings_path = verify_path(settings_path, suffix=".toml")
            with settings_path.open("rb") as f:
                settings = tomllib.load(f)

        if _from_existing:
            # This ensures that the actual settings are being overwritten
            default_settings = settings
        else:
            settings_default_path = verify_path("settings/default.toml")
            with settings_default_path.open("rb") as f:
                default_settings = tomllib.load(f)
            if default_settings.get('gammaloop', {}).get('state_dir', "default") == "default":
                raise ValueError(
                    """No default gammaloop state directory specified. 
                    You can set it using 'glnis setdef -d <path_to_gammaloop_examples>'."""
                )

        for t in settings.get("templates", []):
            try:
                template = verify_path(t, suffix=".toml")
                with template.open("rb") as f:
                    template = tomllib.load(f)
            except (FileNotFoundError, OSError, TypeError):
                try:
                    template = tomllib.loads(t)
                except tomllib.TOMLDecodeError:
                    raise ValueError(
                        f"Template '{t}' is neither a valid path to a .toml file nor a valid TOML string.")
            default_settings = overwrite_settings(
                default_settings, template)
        # if _from_existing, default_settings = settings and hence this overwrite does nothing
        self.settings: Dict[str, Any] = overwrite_settings(
            default_settings, settings,
            always_overwrite=['layered_parameterisation', 'templates'])

        self.gammaloop_state_path = Path(self.settings['gammaloop']['state_dir'],
                                         self.settings['gammaloop']['state'])
        self.settings['integrand']['gammaloop'][
            'gammaloop_state_path'] = str(self.gammaloop_state_path)
        self._graph_from_state = self.settings['graph']['from_state']
        if self._graph_from_state:
            import gammaloop
            self.gammaloop_state = gammaloop.GammaLoopAPI(
                self.gammaloop_state_path,
                level=gammaloop.LogLevel.Off,
                logfile_level=gammaloop.LogLevel.Off,
                read_only_state=True)
            self.dot_path = "Loaded dot from state."
        else:
            self.gammaloop_state = None
            self.dot_path = Path(self.settings['graph']['dot_path'])
        if self.settings['model']['from_state']:
            self.model_path = "Loaded model from state."
        else:
            self.model_path = self.settings['model']['model_path']

    def settings_with_additional_templates(self, template_list: List[str | Path]) -> 'SettingsParser':
        """
        Returns a new SettingsParser with the provided template(s) added to the list of templates in the settings.
        Templates can be provided as paths to .toml files or as valid TOML strings.
        """
        new_settings = deepcopy(self.settings)
        if not isinstance(template_list, list):
            template_list = [template_list]
        new_settings['templates'].extend(template_list)
        return SettingsParser(new_settings, _from_existing=True)

    def get_gammaloop_integration_result(self) -> Dict | None:
        result_path = Path(self.settings['gammaloop']['state_dir'],
                           self.settings['gammaloop']['integration_workspace'],
                           self.settings['gammaloop']['result_file'])
        if not result_path.exists():
            return None

        with result_path.open("r") as f:
            gammaloop_result = json.load(f)

        return gammaloop_result

    def get_integration_target(self) -> IntegrationResult:
        gammaloop_result = self.get_gammaloop_integration_result()
        if gammaloop_result is None:
            return IntegrationResult(**self.settings.get('integration_target', {}))
        try:
            itg_name = self.settings['integrand']['gammaloop']['integrand_name']
            slots = gammaloop_result['slots']
            candidates = [slot for slot in slots
                          if slot['integrand'] == itg_name]
            if len(candidates) == 0:
                raise ValueError("No matching slots found in GammaLoop result.")
            candidate = candidates[0]
            target = candidate.get('target') or candidate.get('integral')

            return IntegrationResult(
                n_points=target['neval'],
                real_central_value=target['result']['re'],
                imag_central_value=target['result']['im'],
                real_error=target['error']['re'],
                imag_error=target['error']['im'],
            )
        except:
            return IntegrationResult(**self.settings.get('integration_target', {}))

    def get_model(self) -> ModelParser:
        return ModelParser(self.model_path)

    def get_parameterisation_kwargs(self) -> List[Dict[str, Any]]:
        nested_kwargs: Dict[str, Any] = deepcopy(
            self.settings['layered_parameterisation'])
        kwargs_list = []
        for new_kwargs in nested_kwargs.values():
            param_type = new_kwargs['parameterisation_type']
            old_kwargs = self.settings['parameterisation'].get(param_type, {})
            # old_kwargs['parameterisation_type'] = param_type
            kwargs_list.append(overwrite_settings(old_kwargs, new_kwargs))

        return kwargs_list

    def get_integrand_kwargs(self) -> Dict[str, Any]:
        new_kwargs = self.settings['layered_integrand']
        new_kwargs['target'] = self.get_integration_target()
        old_kwargs = deepcopy(self.settings['integrand'].get(new_kwargs['integrand_type'], {}))

        return overwrite_settings(old_kwargs, new_kwargs)

    def get_integrator_kwargs(self) -> Dict[str, Any]:
        new_kwargs = self.settings['layered_integrator']
        old_kwargs = deepcopy(self.settings['integrator'].get(new_kwargs['integrator_type'], {}))

        return overwrite_settings(old_kwargs, new_kwargs)

    def get_graph_properties(self) -> GraphProperties | List[GraphProperties]:
        if self.settings['graph']['overwrite_graph_properties']:
            return GraphProperties(**self.settings['graph']['graph_properties'])

        if self._graph_from_state:
            process_id = self.settings['integrand']['gammaloop']['process_id']
            integrand_name = self.settings['integrand']['gammaloop']['integrand_name']
            outputs = dict()
            for o in self.gammaloop_state.list_outputs():
                outputs.update(o)
            if integrand_name not in outputs:
                integrand_name = list(outputs)[0]
            process_id = outputs[integrand_name]
            iinfo = self.gammaloop_state.get_integrand_info(process_id, integrand_name)
            model_as_str = self.gammaloop_state.get_model()
            Model = ModelParser(model_as_str, from_string=True)
            dot_as_str = self.gammaloop_state.get_dot_files(process_id, integrand_name)
            Dot = DotParser(dot_as_str, Model, dot_from_string=True)
            kinematics = self.gammaloop_state.get_default_runtime_settings().kinematics
            e_cm = kinematics.e_cm
            ext_momenta = kinematics.externals.data.momenta.to_dict()
            # Gammaloop indexes the externals before the internals
            n_externals = len(ext_momenta)
            graph_properties_list = []
            for g_id, graph_group in enumerate(iinfo.graph_groups):
                master_id = [g.graph_id for g in graph_group.graphs if g.is_master][0]
                graph_properties = Dot.get_graph_properties(master_id, ext_momenta)
                lmbs = graph_group.loop_momentum_bases
                momentum_space = self.settings['integrand']['gammaloop']['momentum_space']
                if self.settings['graph'].get('overwrite_lmb_heuristics', False) and momentum_space:
                    active_lmbs = lmbs
                else:
                    active_lmbs = [lmb for lmb in lmbs if lmb.channel_id is not None]
                generation_basis_id = [lmb.matches_generation_basis for lmb in lmbs].index(True)
                generation_channel_id = active_lmbs.index(lmbs[generation_basis_id])
                graph_properties.lmb_array = [[e_id-n_externals for e_id in lmb.edge_ids] for lmb in active_lmbs]
                graph_properties.orientation_ids = [o.orientation_id for o in graph_group.orientations]
                graph_properties.orientation_signatures = [o.signature for o in graph_group.orientations]
                graph_properties.generation_channel_id = generation_channel_id
                graph_properties.e_cm = e_cm
                graph_properties.__post_init__()

                n_int_edges = graph_properties.n_edges
                edge_weight = self.settings['graph']['momtrop_edge_weight']
                match edge_weight:
                    case int() | float():
                        edge_weight = n_int_edges*[float(edge_weight)]
                    case [_, *_]:
                        if not len(edge_weight) == n_int_edges:
                            raise ValueError("If provided as a sequence, the number of momtrop "
                                             + "edgeweights must match the number of internal edges.")
                        edge_weight = edge_weight
                    case "default":
                        default_weight = (
                            3*graph_properties.n_loops + 3/2)/n_int_edges/2
                        edge_weight = n_int_edges*[default_weight]
                    case _:
                        raise ValueError("Momtrop edge weights must be one of: \n"
                                         + "Number, Sequence of Numbers or \"default\".")

                graph_properties.momtrop_edge_weight = edge_weight
                graph_properties_list.append(graph_properties)

            return graph_properties_list

        Dot = DotParser(self.dot_path, self.model_path)
        ext_momenta = self.settings['graph']['external_momenta']
        graph_properties = Dot.get_graph_properties(0, ext_momenta)
        graph_properties.generation_channel_id = 0
        graph_properties.e_cm = 0.0
        graph_properties.__post_init__()

        n_int_edges = graph_properties.n_edges

        edge_weight = self.settings['graph']['momtrop_edge_weight']
        match edge_weight:
            case int() | float():
                edge_weight = n_int_edges*[float(edge_weight)]
            case [_, *_]:
                if not len(edge_weight) == n_int_edges:
                    raise ValueError("If provided as a sequence, the number of momtrop "
                                     + "edgeweights must match the number of internal edges.")
                edge_weight = edge_weight
            case "default":
                default_weight = (
                    3*graph_properties.n_loops + 3/2)/n_int_edges/2
                edge_weight = n_int_edges*[default_weight]
            case _:
                raise ValueError("Momtrop edge weights must be one of: \n"
                                 + "Number, Sequence of Numbers or \"default\".")

        graph_properties.momtrop_edge_weight = edge_weight

        return graph_properties

    @staticmethod
    def momtrop_sampler_from_graph_properties(
            gp: GraphProperties) -> Tuple[momtrop.Sampler, momtrop.EdgeData, momtrop.Settings]:

        mt_edges = [momtrop.Edge(tuple(src_dst), ismassive, weight) for src_dst, ismassive, weight
                    in zip(gp.edge_src_dst_vertices, gp.edge_ismassive, gp.momtrop_edge_weight)]
        assym_graph = momtrop.Graph(mt_edges, gp.graph_external_vertices)
        momentum_shifts = [momtrop.Vector(*shift)
                           for shift in gp.edge_momentum_shifts]
        sampler = momtrop.Sampler(assym_graph, gp.graph_signature)
        edge_data = momtrop.EdgeData(gp.edge_masses, momentum_shifts)
        sampler_settings = momtrop.Settings(False, False)

        return sampler, edge_data, sampler_settings
