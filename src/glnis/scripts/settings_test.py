# type: ignore
import torch
import signal
from time import time
from dataclasses import asdict

from glnis.core.integrator import Integrator
from glnis.core.parser import SettingsParser
from glnis.utils.helpers import shell_print


def run_settings_test(file: str, show_graph_properties: bool = False) -> None:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        Settings = SettingsParser(file)

        # Initialize the integrand and integrator
        torch.set_default_dtype(torch.float64)

        time_last = time()
        integrator = Integrator.from_settings(Settings.settings)
        shell_print(f"""Initializing the Integrand and Integrator took {
            - time_last + (time_last := time()):.2f}s""")

        if show_graph_properties:
            from glnis.utils.helpers import Colour
            shell_print("Graph properties:")
            for key, value in asdict(integrator.integrand.param.param.graph_properties).items():
                shell_print(f"{Colour.CYAN}{key}{Colour.END}: {value}")

        # Training parameters
        nitn = 1
        batch_size = 10_000

        shell_print("Attempting training step.")
        integrator.train(nitn, batch_size)
        shell_print(f"Training step successfully completed!")

        shell_print("Attempting integration.")
        integrator.integrate(batch_size)
        shell_print(f"Test successfully completed!")

    except KeyboardInterrupt:
        shell_print(f"\nCaught KeyboardInterrupt — stopping workers: {e}")
        integrator.free()
    except Exception as e:
        shell_print(f"\nCaught Exception — stopping workers: {e}")
        from traceback import print_exc
        print_exc()
        integrator.free()
        raise
    finally:
        integrator.free()
