# type: ignore
import torch
import math
import signal
from time import time

from glnis.core.integrator import Integrator
from glnis.core.parser import SettingsParser
from glnis.utils.helpers import shell_print


def run_state_test(file: str) -> None:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        Settings = SettingsParser(file)
        if not Settings.gammaloop_state_path.exists():
            raise NotADirectoryError(
                f"""No GammaLoop state at {Settings.gammaloop_state_path}""")

        # Initialize the integrand and integrator
        torch.set_default_dtype(torch.float64)

        time_last = time()
        integrator = Integrator.from_settings(Settings.settings)
        shell_print(f"""Initializing the Integrand and Integrator took {
            - time_last + (time_last := time()):.2f}s""")

        # Training parameters
        nitn = 1
        batch_size = 100_000

        # Parse GammaLoop results
        gl_res = Settings.get_gammaloop_integration_result()
        if gl_res is not None:
            RE_OR_IM = 're' if integrator.integrand.training_phase == 'real' else 'im'
            gl_int = gl_res['result'][RE_OR_IM]
            gl_err = gl_res['error'][RE_OR_IM]
            gl_rsd = abs(gl_err / gl_int) * math.sqrt(gl_res['neval'])

            shell_print(f"""Gammaloop Result:    {
                gl_int:.8g} +- {gl_err:.8g}, RSD = {gl_rsd:.2f}""")

        shell_print("Attempting training step.")
        integrator.train(nitn, batch_size)

        shell_print("Attempting integration.")
        time_last = time()
        output = integrator.integrate(batch_size)
        shell_print(f"""Evaluating {batch_size} samples using {integrator.integrand.n_cores} cores took {
            - time_last + (time_last := time()):.2f}s""")
        shell_print(output.str_report())

        shell_print(f"Test successfully completed!")
        shell_print(
            f"The gammaloop state specified in {file} should be good to go.")

    except KeyboardInterrupt:
        shell_print(f"\nCaught KeyboardInterrupt — stopping workers: {e}")
        integrator.integrand.end()
    except Exception as e:
        shell_print(f"\nCaught Exception — stopping workers: {e}")
        integrator.integrand.end()
    finally:
        integrator.integrand.end()
