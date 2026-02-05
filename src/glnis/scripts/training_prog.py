# type: ignore
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
from glnis.core.integrator import MadnisIntegrator, Integrator, NaiveIntegrator
from glnis.utils.helpers import error_fmter


def run_training_prog(settings_file: str,
                      comment: str = "",
                      no_output: bool = False,
                      ) -> None:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        print(f"| > Working on settings {settings_file}")
        Settings = SettingsParser(settings_file)

        if not no_output:
            PROJECT_ROOT = Path(__file__).parents[3]
            subfolder_path = Path(
                PROJECT_ROOT, "outputs", "training_prog")
            print(f"| > Output will be at {subfolder_path}")

        # Training parameters
        torch.set_default_dtype(torch.float64)
        time_last = time()
        params = Settings.settings['scripts']['training_prog']
        n_training_steps = params['n_training_steps']
        n_log = params['n_log']
        n_plot_rsd = params['n_plot_rsd']
        n_plot_loss = params['n_plot_loss']
        n_samples = params['n_samples']
        n_samples_after_training = params['n_samples_after_training']
        gammaloop_state = Settings.settings['gammaloop_state']['state_name']

        integrator: MadnisIntegrator = Integrator.from_settings_file(
            settings_file)
        naive_integrator = NaiveIntegrator(integrator.integrand, seed=42)

        # Plotting setup
        losses = []
        rsds = []
        steps_losses = []
        steps_rsds = []

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
                print(f"""| > Trained Result after {step} steps of {integrator.batch_size}: {
                    metrics.integral:.8g} +- {metrics.error:.8g}, RSD = {rsd:.3f}""")
                rsds.append(rsd)
                steps_rsds.append(step)

        integrator.callback = callback

        print(f"""| > Initializing the Integrand and Integrator took {
            - time_last + (time_last := time()):.2f}s""")

        # Parse GammaLoop results
        gl_res = Settings.get_gammaloop_integration_result()
        if gl_res is not None:
            RE_OR_IM = 're' if integrator.integrand.training_phase == 'real' else 'im'
            gl_int = gl_res['result'][RE_OR_IM]
            gl_err = gl_res['error'][RE_OR_IM]
            gl_neval = gl_res['neval']
            gl_rsd = abs(gl_err / gl_int) * math.sqrt(gl_neval)

            print(
                f"| > Gammaloop Result: {gl_int:.8g} +- {gl_err:.8g}, RSD = {gl_rsd:.3f}")

        time_last = time()
        naive_output = naive_integrator.integrate(n_samples)
        print(f"""| > Evaluating {n_samples} samples using Naive integrator and {naive_integrator.integrand.n_cores} cores took {
            - time_last + (time_last := time()):.2f}s""")

        print(naive_output.str_report())

        integrator.train(n_training_steps, callback)

        if gl_res is not None:
            print(
                f"| > Gammaloop Result: {gl_int:.8g} +- {gl_err:.8g}, RSD = {gl_rsd:.3f}")

        # Take the final snapshot
        metrics = integrator.integrate(n_samples_after_training)
        trained_int = metrics.integral
        trained_err = metrics.error
        trained_rsd = metrics.rel_stddev
        print(f"""| > Trained Result after {n_training_steps} steps of {integrator.batch_size}, using a sample size of {n_samples_after_training}: {
            trained_int:.8g} +- {trained_err:.8g}, RSD = {trained_rsd:.3f}""")

        # IMPORTANT: close the worker functions, or your script will hang
        integrator.integrand.end()

        if no_output:
            quit()

        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
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
            if gl_res is not None:
                f.write(f"\n{line}")
                f.write(f"{' Gammaloop Results ':{'#'}^{width}}\n")
                f.write(line)
                f.write(f"Integral: {error_fmter(gl_int, gl_err)}\n")
                f.write(f"RSD: {gl_rsd:.3f}\n")
                f.write(f"Number of samples: {gl_neval}\n")
            f.write(f"\n{line}")
            f.write(f"{' Momtrop Results ':{'#'}^{width}}\n")
            f.write(line)
            f.write(f"Integral: {error_fmter(momtrop_int, momtrop_err)}\n")
            f.write(f"RSD: {momtrop_rsd:.3f}\n")
            f.write(f"Number of samples: {n_samples}\n")
            f.write(f"\n{line}")
            f.write(f"{' Trained Results ':{'#'}^{width}}\n")
            f.write(line)
            f.write(f"Integral: {error_fmter(trained_int, trained_err)}\n")
            f.write(f"RSD: {trained_rsd:.3f}\n")
            f.write(f"Number of samples: {n_samples_after_training}\n")
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt — stopping workers.")
        integrator.integrand.end()
    finally:
        integrator.integrand.end()
