# type: ignore
import sys


def par_spl_comp(settings_file: str,
                 comment: str = "",
                 par_idx: int = 0
                 ) -> None:
    par_idx = int(par_idx)
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    import signal
    import os
    from pathlib import Path
    from datetime import datetime
    from time import time

    from glnis.core.parser import SettingsParser
    from glnis.core.integrator import MadnisIntegrator, VegasIntegrator, NaiveIntegrator
    from glnis.core.integrand import MPIntegrand
    from glnis.utils.helpers import error_fmter

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        print(f"| > Working on settings {settings_file}")
        Settings = SettingsParser(settings_file)
        graph_properties = Settings.get_graph_properties()
        graph_properties.lmb_array = np.arange(
            6, dtype=np.uint64).reshape(-1, 1)
        gammaloop_state = Settings.settings['gammaloop_state']['state_name']

        PROJECT_ROOT = Path(__file__).parents[1]
        subfolder_path = Path(
            PROJECT_ROOT, "outputs", "par_spl_comp")
        print(f"| > Output will be at {subfolder_path}")
        filename = gammaloop_state+datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

        # Training settings
        n_cores = 64
        n_madnis_train = 2000
        n_log = 20
        n_samples = 10_000_000
        n_vegas_train = 10
        vegas_batch_size = 100_000

        # Integrand stays the same, as does the seed for the naive integrator
        integrand_kwargs = Settings.get_integrand_kwargs()
        integrand_kwargs.pop('n_cores')
        integrand_kwargs.pop('verbose')
        gl_result = Settings.get_gammaloop_integration_result()
        if gl_result is not None:
            target_real_mean = gl_result['result']['re']
            target_imag_mean = gl_result['result']['im']
            target_real_error = gl_result['error']['re']
            target_imag_error = gl_result['error']['im']
            if target_real_mean == 0.:
                target_real_rsd = 0.
            else:
                target_real_rsd = target_real_error / \
                    target_real_mean * math.sqrt(gl_result['neval'])
            if target_imag_mean == 0.:
                target_imag_rsd = 0.
            else:
                target_imag_rsd = target_imag_error / \
                    target_imag_mean * math.sqrt(gl_result['neval'])
            integrand_kwargs['target_real'] = target_real_mean
            integrand_kwargs['target_imag'] = target_imag_mean
        naive_seed = 42
        """ if integrand_kwargs['integrand_type'] == "kaapo":
            norm_factor = (2*math.pi)**-(3*graph_properties.n_loops)
        else:
            norm_factor = 1. """

        # Let the spaghetti begin
        # Integrator settings for all type
        Settings.settings['layered_integrator'] = dict(
            integrator_type="naive", seed=naive_seed)
        naive_kwargs = Settings.get_integrator_kwargs()
        naive_kwargs.pop('integrator_type')
        Settings.settings['layered_integrator'] = dict(integrator_type="vegas")
        vegas_kwargs = Settings.get_integrator_kwargs()
        vegas_kwargs.pop('integrator_type')

        def madnis_callback(status) -> None:
            step = status.step + 1
            if step % n_log == 0:
                print(
                    f"| > Step {step}: loss={status.loss:.5f}, lr={status.learning_rate:.2e}")

        # Param settings
        Settings.settings['layered_parameterisation'] = dict(
            layer_0=dict(parameterisation_type="spherical")
        )
        spherical_kwargs = Settings.get_parameterisation_kwargs()
        Settings.settings['layered_parameterisation'] = dict(
            layer_0=dict(parameterisation_type="spherical"),
            layer_1=dict(parameterisation_type="mc_layer",
                         subtype="ose",
                         ose_exponent=2.0,),
        )
        mc_spherical_kwargs = Settings.get_parameterisation_kwargs()
        Settings.settings['layered_parameterisation'] = dict(
            layer_0=dict(parameterisation_type="momtrop",
                         sample_discrete=False)
        )
        momtrop_nodisc_kwargs = Settings.get_parameterisation_kwargs()
        Settings.settings['layered_parameterisation'] = dict(
            layer_0=dict(parameterisation_type="momtrop",
                         sample_discrete=True)
        )
        momtrop_kwargs = Settings.get_parameterisation_kwargs()

        param_kwargs_list = [
            spherical_kwargs, mc_spherical_kwargs, momtrop_nodisc_kwargs, momtrop_kwargs
        ]
        param_type_list = [
            "spherical", "mc_spherical", "momtrop_nodisc", "momtrop"
        ]

        naive_results = dict(
            real_means=[], real_errors=[], real_rsd=[
            ], imag_means=[], imag_errors=[], imag_rsd=[],
        )
        vegas_results = dict(
            real_means=[], real_errors=[], real_rsd=[
            ], imag_means=[], imag_errors=[], imag_rsd=[],
        )
        madnis_results = dict(
            means=[], errors=[], rsd=[],
        )
        param_type = param_type_list[par_idx]
        param_kwargs = param_kwargs_list[par_idx]

        Settings.settings['layered_integrator'] = dict(
            integrator_type="madnis", use_scheduler=True, n_train_for_scheduler=n_madnis_train)
        madnis_kwargs = Settings.get_integrator_kwargs()
        """ madnis_kwargs = dict(
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
                 ),
                 callback: Callable[[object], None] | None = None,
        ) """
        madnis_kwargs.pop('integrator_type')
        time_last = time()
        integrand = MPIntegrand(
            graph_properties=graph_properties,
            param_kwargs=param_kwargs,
            integrand_kwargs=integrand_kwargs,
            n_cores=n_cores,
        )
        print(f"""| > Initializing the Integrand with param '{param_type}' took {
            - time_last + (time_last := time()):.2f}s""")
        print(
            f"""| > Integrand has cdim: {integrand.continuous_dim} and ddim: {integrand.discrete_dims}.""")

        print(f"""| > STARTING NAIVE INTEGRATION""")
        naive_integrator = NaiveIntegrator(integrand, **naive_kwargs)
        time_last = time()
        output = naive_integrator.integrate(n_samples)
        print(f"""| > Evaluating {n_samples} samples using {naive_integrator.integrand.n_cores} cores took {
            - time_last + (time_last := time()):.2f}s""")
        naive_obs = output.get_observables()
        naive_real_mean = naive_obs['real_central_value']
        naive_real_error = naive_obs['real_error']
        naive_results['real_means'].append(naive_real_mean)
        naive_results['real_errors'].append(naive_real_error)
        if naive_real_mean == 0.:
            naive_real_rsd = 0.
        else:
            naive_real_rsd = abs(
                naive_real_error / naive_real_mean) * math.sqrt(output.modules[0].n_points)
        naive_results['real_rsd'].append(naive_real_rsd)
        naive_imag_mean = naive_obs['imag_central_value']
        naive_imag_error = naive_obs['imag_error']
        naive_results['imag_means'].append(naive_imag_mean)
        naive_results['imag_errors'].append(naive_imag_error)
        if naive_imag_mean == 0.:
            naive_imag_rsd = 0.
        else:
            naive_imag_rsd = abs(
                naive_imag_error / naive_imag_mean) * math.sqrt(output.modules[0].n_points)
        naive_results['imag_rsd'].append(naive_imag_rsd)
        print(output.str_report())

        print(f"""| > STARTING VEGAS TRAINING""")
        vegas_integrator = VegasIntegrator(integrand, **vegas_kwargs)
        vegas_report = vegas_integrator.train(
            n_vegas_train, vegas_batch_size)
        print(vegas_report.summary())
        print(f"""| > STARTING VEGAS INTEGRATION""")
        time_last = time()
        output = vegas_integrator.integrate(n_samples)
        print(f"""| > Evaluating {n_samples} samples using {vegas_integrator.integrand.n_cores} cores took {
            - time_last + (time_last := time()):.2f}s""")
        vegas_obs = output.get_observables()
        vegas_real_mean = vegas_obs['real_central_value']
        vegas_real_error = vegas_obs['real_error']
        vegas_results['real_means'].append(vegas_real_mean)
        vegas_results['real_errors'].append(vegas_real_error)
        if vegas_real_mean == 0.:
            vegas_real_rsd = 0.
        else:
            vegas_real_rsd = abs(
                vegas_real_error / vegas_real_mean) * math.sqrt(output.modules[0].n_points)
        vegas_results['real_rsd'].append(vegas_real_rsd)
        vegas_imag_mean = vegas_obs['imag_central_value']
        vegas_imag_error = vegas_obs['imag_error']
        vegas_results['imag_means'].append(vegas_imag_mean)
        vegas_results['imag_errors'].append(vegas_imag_error)
        if vegas_imag_mean == 0.:
            vegas_imag_rsd = 0.
        else:
            vegas_imag_rsd = abs(
                vegas_imag_error / vegas_imag_mean) * math.sqrt(output.modules[0].n_points)
        vegas_results['imag_rsd'].append(vegas_imag_rsd)
        print(output.str_report())

        print(f"""| > STARTING MADNIS TRAINING""")
        madnis_integrator = MadnisIntegrator(
            integrand=integrand,
            integrator_kwargs=madnis_kwargs,
            callback=madnis_callback,
        )
        madnis_integrator.train(n_madnis_train)
        print(f"""| > STARTING MADNIS INTEGRATION""")
        time_last = time()
        output = madnis_integrator.integrate(n_samples)
        print(f"""| > Evaluating {n_samples} samples using {madnis_integrator.integrand.n_cores} cores took {
            - time_last + (time_last := time()):.2f}s""")
        print(
            f"| > MadNIS result after {n_madnis_train} steps of batch size {madnis_integrator.batch_size}:")
        print(
            f"| >      {output.integral:.8e} +- {output.error:.8e}, RSD={output.rel_stddev:.3f}")
        madnis_results['means'].append(output.integral)
        madnis_results['errors'].append(output.error)
        madnis_results['rsd'].append(output.rel_stddev)

        integrand.end()

        with Path(subfolder_path, filename+".txt").open('w') as f:
            sep = '-'
            width = 60
            line = width*sep+"\n"
            f.write(f"Comment: {comment} \n")
            f.write(f"Gammaloop State: {gammaloop_state}\n")
            f.write(f"Gammaloop Results: \n")
            f.write(
                f"  RE: {target_real_mean:.8e} +- {target_real_error:.8e}, RSD={target_real_rsd}\n")
            f.write(
                f"  IM: {target_imag_mean:.8e} +- {target_imag_error:.8e}, RSD={target_imag_rsd}\n")
            f.write(line)
            f.write(f"{' Training Parameters ':{'#'}^{width}}\n")
            f.write(line)
            f.write(
                f"Trained on phase: {madnis_integrator.integrand.training_phase}\n")
            f.write(f"Number of samples: {n_samples}\n")
            f.write(f"MadNIS Batch Size: {madnis_integrator.batch_size}\n")
            f.write(f"Number of MadNIS training steps: {n_madnis_train}\n")
            discrete_model = Settings.settings['integrator']['madnis']['discrete_model']
            f.write(
                f"Discrete Model: {discrete_model}\n")
            discrete_model_kwargs = Settings.settings['integrator']['madnis'][discrete_model]
            f.write(f"Discrete model kwargs: \n")
            for k, v in discrete_model_kwargs.items():
                f.write(f"  {k}: {v} \n")

            f.write(f"Parameterisation: {param_type}")
            f.write(f"\n{line}")
            f.write(f"{' Untrained Results ':{'#'}^{width}}\n")
            f.write(line)
            f.write(f"  RE MEAN : {float(naive_results['real_means'][0])} \n")
            f.write(f"  RE ERROR: {float(naive_results['real_errors'][0])} \n")
            f.write(f"  RE RSD  : {float(naive_results['real_rsd'][0])} \n")
            f.write(f"  IM MEAN : {float(naive_results['imag_means'][0])} \n")
            f.write(f"  IM ERROR: {float(naive_results['imag_errors'][0])} \n")
            f.write(f"  IM RSD  : {float(naive_results['imag_rsd'][0])} \n")
            f.write(f"\n{line}")
            f.write(f"{' Vegas Results ':{'#'}^{width}}\n")
            f.write(line)
            f.write(f"  RE MEAN : {float(vegas_results['real_means'][0])} \n")
            f.write(f"  RE ERROR: {float(vegas_results['real_errors'][0])} \n")
            f.write(f"  RE RSD  : {float(vegas_results['real_rsd'][0])} \n")
            f.write(f"  IM MEAN : {float(vegas_results['imag_means'][0])} \n")
            f.write(f"  IM ERROR: {float(vegas_results['imag_errors'][0])} \n")
            f.write(f"  IM RSD  : {float(vegas_results['imag_rsd'][0])} \n")
            f.write(f"{' MadNIS Results ':{'#'}^{width}}\n")
            f.write(line)
            f.write(f"  MEAN : {float(madnis_results['means'][0])} \n")
            f.write(f"  ERROR: {float(madnis_results['errors'][0])} \n")
            f.write(f"  RSD  : {float(madnis_results['rsd'][0])} \n")
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt — stopping workers.")
        integrand.end()
    finally:
        integrand.end()


if __name__ == "__main__":
    settings_file = sys.argv[1]
    par_idx = sys.argv[2]
    par_spl_comp(settings_file, "", par_idx)
