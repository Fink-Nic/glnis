# type: ignore
import torch
import math
import signal
from time import time

from glnis.core.integrator import Integrator
from glnis.core.parser import SettingsParser


def run_state_test(settings_file: str) -> None:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        Settings = SettingsParser(settings_file)
        if not Settings.gammaloop_state_path.exists():
            raise NotADirectoryError(
                f"""No GammaLoop state at {Settings.gammaloop_state_path}""")

        # Initialize the gammaloop integrand and madnis integrator
        torch.set_default_dtype(torch.float64)

        time_last = time()
        integrator = Integrator.from_settings_file(settings_file)
        print(f"""| > Initializing the Integrand and Integrator took {
            - time_last + (time_last := time()):.2f}s""")

        # Training parameters
        n_training_steps = 1
        batch_size = 100_000

        # Parse GammaLoop results
        gl_res = Settings.get_gammaloop_integration_result()
        if gl_res is not None:
            RE_OR_IM = 're' if integrator.integrand.training_phase == 'real' else 'im'
            gl_int = gl_res['result'][RE_OR_IM]
            gl_err = gl_res['error'][RE_OR_IM]
            gl_rsd = abs(gl_err / gl_int) * math.sqrt(gl_res['neval'])

            print(f"""| > Gammaloop Result:    {
                gl_int:.8g} +- {gl_err:.8g}, RSD = {gl_rsd:.2f}""")

        time_last = time()
        output = integrator.integrate(batch_size)
        print(f"""| > Evaluating {batch_size} samples using {integrator.integrand.n_cores} cores took {
            - time_last + (time_last := time()):.2f}s""")

        if integrator.IDENTIFIER.lower() == 'madnis sampler':
            print(f"""| > Result (before training) using {batch_size} samples:     {
                output.integral:.8g} +- {output.error:.8g}, RSD = {output.rel_stddev:.2f}""")
            print("| > "+100*"=")
            print("| > Attempting training step.")
            integrator.train(n_training_steps)
        else:
            print(output.str_report())
            print("| > "+100*"=")
            print("| > Attempting training step.")
            integrator.train(n_training_steps, batch_size=1_000)

        print(f"| > Test successfully completed!")
        print(
            f"| > The gammaloop state specified in {settings_file} should be good to go.")
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt — stopping workers.")
        integrator.integrand.end()
    finally:
        integrator.integrand.end()
