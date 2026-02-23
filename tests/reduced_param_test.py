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
    from glnis.core.integrator import MadnisIntegrator, Integrator, NaiveIntegrator, VegasIntegrator
    from glnis.utils.helpers import error_fmter

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        torch.set_default_dtype(torch.float64)
        print(f"| > Working on settings {settings_file}")
        Settings = SettingsParser(settings_file)
        graph_properties = Settings.get_graph_properties()

        norm_factor = (2*math.pi)**-(3*graph_properties.n_loops)

        integrator: MadnisIntegrator = Integrator.from_settings_file(
            settings_file)
        integrand = integrator.integrand
        parameterisation = integrand.param

        input = integrator.init_layer_data(9)
        # input.continuous = np.array([
        #     [0.1, 0.1, 0.1],
        #     [0.2, 0.2, 0.2],
        #     [0.3, 0.3, 0.3],
        #     [0.4, 0.4, 0.4],
        #     [0.5, 0.5, 0.5],
        #     [0.6, 0.6, 0.6],
        #     [0.7, 0.7, 0.7],
        #     [0.8, 0.8, 0.8],
        #     [0.9, 0.9, 0.9],
        # ])
        input.continuous = np.tile(
            (np.arange(9)/10.+0.1).reshape(-1, 1), (1, integrand.continuous_dim))
        input.discrete = np.array([
            [2], [2], [2],
            [0], [0], [0],
            [1], [1], [1],
        ], dtype=np.uint64)
        # input.discrete = np.arange(3)
        print(f"Before parameterisation:")
        print(f"{input.continuous=}")
        print(f"{input.discrete=}")
        output = parameterisation.parameterise(input)
        jac = output.jac.flatten()
        momenta = output.momenta.reshape(-1, 2, 3)
        print(f"After Parameterisation:")
        print(f"{momenta=}")
        print(f"{jac=}")
        print(f"Scaled Jacobians: {jac * norm_factor}")

    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt — stopping workers.")
        integrator.integrand.end()
    finally:
        integrator.integrand.end()


if __name__ == "__main__":
    run_training_prog(sys.argv[1], comment="", no_output=True)
