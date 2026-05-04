# type: ignore
import torch
import signal
from time import time

from glnis.core.integrator import Integrator
from glnis.core.parser import SettingsParser
from glnis.utils.helpers import shell_print, Colour


def run_settings_test(file: str, show_graph_properties: bool = False) -> None:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        Settings = SettingsParser(file)

        if show_graph_properties:
            gps = Settings.get_graph_properties()
            if not isinstance(gps, list):
                gps = [gps]
            for i, gp in enumerate(gps):
                shell_print(f"Graph properties of graph {Colour.PURPLE}{i}{Colour.END}:")
                for key, value in gp.__dict__.items():
                    shell_print(f"    {Colour.CYAN}{key}{Colour.END}: {value}")

        # Initialize the integrand and integrator
        torch.set_default_dtype(torch.float64)

        time_last = time()
        integrator = Integrator.from_settings(Settings.settings)
        shell_print(f"""Initializing the Integrand and Integrator took {Colour.CYAN}{
            - time_last + (time_last := time()):.2f}s{Colour.END}""")

        integrator.display_info()

        # Training parameters
        nitn = 1
        batch_size = 10_000

        shell_print("Attempting training step.")
        integrator.train(nitn, batch_size)
        shell_print(f"Training step successfully completed!")

        shell_print("Attempting integration.")
        integrator.integrate(batch_size)
        shell_print(f"{Colour.GREEN}Test successfully completed!{Colour.END}")

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
