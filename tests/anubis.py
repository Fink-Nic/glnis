# type: ignore
import sys


def run_training_prog(settings_file: str,
                      comment: str = "",
                      no_output: bool = False,
                      ) -> None:

    import torch
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    import signal
    import os
    from pathlib import Path
    from datetime import datetime
    from time import time

    from glnis.core.parser import SettingsParser
    from glnis.core.integrator import MadnisIntegrator, Integrator, NaiveIntegrator, VegasIntegrator, HavanaIntegrator
    from glnis.utils.helpers import error_fmter

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        torch.set_default_dtype(torch.float64)
        print(f"| > Working on settings {settings_file}")
        Settings = SettingsParser(settings_file)

        # Training parameters
        params = Settings.settings['scripts']['training_prog']
        n_training_steps = params['n_training_steps']
        n_total_training_samples = n_training_steps * \
            Settings.settings['integrator']['madnis']['batch_size']
        n_log = params['n_log']
        n_plot_rsd = params['n_plot_rsd']
        n_plot_loss = params['n_plot_loss']
        n_samples = params['n_samples']
        n_samples_after_training = params['n_samples_after_training']
        nitn = 10  # number of vegas training iterations
        # number of evals per vegas training iteration
        neval = int(n_total_training_samples / nitn / 2)
        gammaloop_state = Settings.settings['gammaloop_state']['state_name']
        graph_properties = Settings.get_graph_properties()
        Settings.settings['integrator']['madnis']['n_train_for_scheduler'] = n_training_steps

        norm_factor = (2*math.pi)**-(3*graph_properties.n_loops)

        if not no_output:
            PROJECT_ROOT = Path(__file__).parents[1]
            subfolder_path = Path(
                PROJECT_ROOT, "outputs", "kaapos")
            print(f"| > Output will be at {subfolder_path}")

        time_last = time()
        integrator: MadnisIntegrator = Integrator.from_settings_file(
            settings_file)
        print(f"| > MadNIS is using device: {integrator.madnis.dummy.device}")
        print(f"| > MadNIS is using scheduler: {integrator.madnis.scheduler}")
        print(
            f"| > Integrand discrete dims: {integrator.integrand.discrete_dims}")

        naive_integrator = NaiveIntegrator(
            integrator.integrand, seed=42)
        vegas_integrator = VegasIntegrator(integrator.integrand)
        havana_integrator = HavanaIntegrator(integrator.integrand, seed=42)

        # Plotting setup
        losses = []
        rsds = []
        steps_losses = []
        steps_rsds = []
        means = []
        errors = []

        # Callback for the madnis integrator
        def callback(status) -> None:
            step = status.step + 1
            if step % n_log == 0:
                print(f"| > Step {status.step + 1}: loss={status.loss:.5f}")
            if step % n_plot_loss == 0:
                losses.append(status.loss)
                steps_losses.append(step)
            if step % n_plot_rsd == 0:
                metrics = integrator.integrate(n_samples)
                rsd = metrics.rel_stddev
                scaled_int = metrics.integral * norm_factor
                scaled_err = metrics.error * norm_factor
                print(f"""| > Trained Result after {step} steps of {integrator.batch_size}, using {
                    n_samples} samples: \n| > {
                    scaled_int:.8e} +- {scaled_err:.8e}, RSD = {rsd:.3f}, lr = {status.learning_rate:.2e}""")
                means.append(scaled_int)
                errors.append(scaled_err)
                rsds.append(rsd)
                steps_rsds.append(step)

        integrator.callback = callback

        # Parse GammaLoop results
        gl_res = Settings.get_gammaloop_integration_result()
        if gl_res is not None:
            norm_factor = 1.
            RE_OR_IM = 're' if integrator.integrand.training_phase == 'real' else 'im'
            gl_int = gl_res['result'][RE_OR_IM]
            gl_err = gl_res['error'][RE_OR_IM]
            gl_neval = gl_res['neval']
            gl_rsd = abs(gl_err / gl_int) * math.sqrt(gl_neval)

            print(
                f"| > Gammaloop Result: {gl_int:.8g} +- {gl_err:.8g}, RSD = {gl_rsd:.3f}")

        if not integrator.integrand.integrand.IDENTIFIER == "kaapo integrand":
            norm_factor = 1.

        print(f"""| > Initializing the Integrand and Integrator took {
            - time_last + (time_last := time()):.2f}s""")

        time_last = time()
        output = naive_integrator.integrate(n_samples_after_training)
        output.modules[0].real_central_value *= norm_factor
        output.modules[0].real_error *= norm_factor
        print(f"""| > Evaluating {n_samples_after_training} samples using {naive_integrator.integrand.n_cores} cores took {
            - time_last + (time_last := time()):.2f}s""")
        untrained_obs = output.get_observables()
        untrained_int = untrained_obs['real_central_value']
        untrained_err = untrained_obs['real_error']
        untrained_rsd = abs(
            untrained_err / untrained_int) * math.sqrt(output.modules[0].n_points)
        print(output.str_report())

        print("| > Training Vegas:")
        vegas_training_result = vegas_integrator.train(nitn, neval)
        print(vegas_training_result.summary())

        print("| > Training Vegas again:")
        vegas_training_result = vegas_integrator.train(nitn, neval)
        print(vegas_training_result.summary())

        time_last = time()
        output = vegas_integrator.integrate(n_samples_after_training)
        output.modules[0].real_central_value *= norm_factor
        output.modules[0].real_error *= norm_factor
        print(f"""| > Evaluating {n_samples_after_training} samples using {vegas_integrator.integrand.n_cores} cores took {
            - time_last + (time_last := time()):.2f}s""")
        vegas_obs = output.get_observables()
        vegas_int = vegas_obs['real_central_value']
        vegas_err = vegas_obs['real_error']
        vegas_rsd = abs(
            vegas_err / vegas_int) * math.sqrt(output.modules[0].n_points)
        print(output.str_report())

        print("| > Training Havana:")
        havana_training_result = havana_integrator.train(2*nitn, neval)
        print(havana_training_result)

        time_last = time()
        output = havana_integrator.integrate(n_samples_after_training)
        output.modules[0].real_central_value *= norm_factor
        output.modules[0].real_error *= norm_factor
        print(f"""| > Evaluating {n_samples_after_training} samples using {havana_integrator.integrand.n_cores} cores took {
            - time_last + (time_last := time()):.2f}s""")
        havana_obs = output.get_observables()
        havana_int = havana_obs['real_central_value']
        havana_err = havana_obs['real_error']
        havana_rsd = abs(
            havana_err / havana_int) * math.sqrt(output.modules[0].n_points)
        print(output.str_report())

        print("| > Training MadNIS:")
        integrator.train(n_training_steps, callback)

        metrics = integrator.integrate(n_samples_after_training)
        rsd = metrics.rel_stddev
        scaled_int = metrics.integral * norm_factor
        scaled_err = metrics.error * norm_factor
        print(f"""| > Trained Result after {n_training_steps} steps of {integrator.batch_size}, using {
            n_samples_after_training} samples: \n| > {
            scaled_int:.8e} +- {scaled_err:.8e}, RSD = {rsd:.3f}""")

        # Print the final snapshot
        madnis_int = scaled_int
        madnis_err = scaled_err
        madnis_rsd = rsd
        print(f"""| > Trained Result after {n_training_steps} steps of {integrator.batch_size}, using a sample size of {n_samples_after_training}: {
            madnis_int:.8g} +- {madnis_err:.8g}, RSD = {madnis_rsd:.3f}""")

        # IMPORTANT: close the worker functions, or your script will hang
        integrator.integrand.end()

        if no_output:
            quit()

        if not os.path.exists(str(subfolder_path)):
            os.makedirs(str(subfolder_path))
        losses, steps_losses = np.array(losses), np.array(steps_losses)
        rsds, steps_rsds = np.array(rsds), np.array(steps_rsds)

        fig, axs = plt.subplots(2, 1, sharex=True, layout="constrained")
        axs: list[plt.Axes]
        axs[0].plot(steps_losses, losses)
        axs[0].set_ylabel("loss")
        axs[1].scatter(steps_rsds, rsds)
        axs[1].set_ylabel("RSD")
        axs[1].set_xlabel("Training steps")
        fig.suptitle(f"Training progression for {gammaloop_state}")
        filename = gammaloop_state+datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        plt.savefig(Path(subfolder_path, filename+".png"),
                    dpi=300, bbox_inches='tight')

        with Path(subfolder_path, filename+".txt").open('w') as f:
            sep = '-'
            width = 60
            line = width*sep+"\n"
            f.write(f"Comment: {comment} \n")
            f.write(line)
            f.write(f"{' Training Parameters ':{'#'}^{width}}\n")
            f.write(line)
            f.write(f"{gammaloop_state=}\n")
            f.write(f"{integrator.batch_size=}\n")
            f.write(f"{n_training_steps=}\n")
            f.write(
                f"Discrete Model: {Settings.settings['integrator']['madnis']['discrete_model']}\n")
            f.write(
                f"Integrated phase: {integrator.integrand.training_phase}\n")
            f.write(f"\n{line}")
            f.write(f"{' Untrained Results ':{'#'}^{width}}\n")
            f.write(line)
            f.write(f"Integral: {untrained_int:.8e} +- {untrained_err:.8e}\n")
            f.write(f"RSD: {untrained_rsd:.3f}\n")
            f.write(f"Number of samples: {n_samples_after_training}\n")
            f.write(f"\n{line}")
            f.write(f"{' Vegas Results ':{'#'}^{width}}\n")
            f.write(line)
            f.write(f"Integral: {vegas_int:.8e} +- {vegas_err:.8e}\n")
            f.write(f"RSD: {vegas_rsd:.3f}\n")
            f.write(f"Number of samples: {n_samples_after_training}\n")
            f.write(f"\n{line}")
            f.write(f"{' Havana Results ':{'#'}^{width}}\n")
            f.write(line)
            f.write(f"Integral: {havana_int:.8e} +- {havana_err:.8e}\n")
            f.write(f"RSD: {havana_rsd:.3f}\n")
            f.write(f"Number of samples: {n_samples_after_training}\n")
            f.write(f"\n{line}")
            f.write(f"{' MadNIS Results ':{'#'}^{width}}\n")
            f.write(line)
            f.write(f"Integral: {madnis_int:.8e} +- {madnis_err:.8e}\n")
            f.write(f"RSD: {madnis_rsd:.3f}\n")
            f.write(f"Number of samples: {n_samples_after_training}\n")
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt — stopping workers.")
        integrator.integrand.end()
    finally:
        integrator.integrand.end()


if __name__ == "__main__":
    run_training_prog(sys.argv[1], comment="", no_output=sys.argv[2])
