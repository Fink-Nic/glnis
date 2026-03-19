# type: ignore

def run_state_import(
    settings_file: str,
    state_file: str,
) -> None:
    from pathlib import Path

    from glnis.utils.helpers import shell_print, verify_path
    from glnis.scripts.sampler_comparison import SamplerCompData

    import signal
    from torch import load

    from glnis.core.integrator import (
        Integrator,
        NaiveIntegrator,
        VegasIntegrator,
        HavanaIntegrator,
        MadnisIntegrator,
    )
    from glnis.core.parser import SettingsParser

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        shell_print(f"Working on settings {settings_file}")
        Settings = SettingsParser(settings_file)

        state_file: Path = verify_path(state_file)
        with state_file.open('rb') as f:
            SData: SamplerCompData = load(f, weights_only=False)

        Settings.settings["layered_integrator"]["integrator_type"] = "madnis"
        madnis_integrator: MadnisIntegrator = Integrator.from_settings_file(
            settings_file
        )
        integrand = madnis_integrator.integrand

        shell_print(f"MadNIS is using device: {madnis_integrator.madnis.dummy.device}")
        shell_print(f"MadNIS is using scheduler: {madnis_integrator.madnis.scheduler}")
        shell_print(f"Integrand discrete dims: {integrand.discrete_dims}")

        # Parse GammaLoop results
        # Creating and importing all the samplers
        for name, state in SData.integrator_states.items():
            shell_print(f"Found {name} state")
            obs = SData.observables.get(name, None)
            if obs is None:
                shell_print(f"Found no observables for {name}. Skipping...")
                continue
            else:
                match name.lower():
                    case "naive":
                        Settings.settings["layered_integrator"]["integrator_type"] = "naive"
                        integrator = NaiveIntegrator(integrand, **Settings.get_integrator_kwargs())
                    case "vegas":
                        Settings.settings["layered_integrator"]["integrator_type"] = "vegas"
                        integrator = VegasIntegrator(integrand, **Settings.get_integrator_kwargs())
                    case "havana":
                        Settings.settings["layered_integrator"]["integrator_type"] = "havana"
                        integrator = HavanaIntegrator(integrand, **Settings.get_integrator_kwargs())
                    case "madnis":
                        integrator = madnis_integrator
                    case _:
                        shell_print(f"Unknown sampler type '{name}' in state file. Skipping...")
                        continue

                n_points = obs.n_points
                shell_print(f"Result before exporting, using {n_points} samples:")
                if obs.real_error:
                    shell_print(
                        f"    RE : {obs.real_central_value:.8e} +- {obs.real_error:.8e}, RSD = {obs.real_rsd:.3f}\n")
                if obs.imag_error:
                    shell_print(
                        f"    IM : {obs.imag_central_value:.8e} +- {obs.imag_error:.8e}, RSD = {obs.imag_rsd:.3f}\n")
            for i in range(2):
                try:
                    integrator.import_state(state)
                except Exception as e:
                    shell_print(f"Failed to import state for {name} with error: {e}. Skipping...")
                    continue
                shell_print(f"Successfully imported {name} state")
                acc = integrator.integrate(n_points, progress_report=False)
                obs = acc.statistics.result
                shell_print(f"Result for {name} after importing, run {i + 1}:")
                if obs.real_error:
                    shell_print(
                        f"    RE : {obs.real_central_value:.8e} +- {obs.real_error:.8e}, RSD = {obs.real_rsd:.3f}\n")
                if obs.imag_error:
                    shell_print(
                        f"    IM : {obs.imag_central_value:.8e} +- {obs.imag_error:.8e}, RSD = {obs.imag_rsd:.3f}\n")

    except KeyboardInterrupt:
        shell_print("\nCaught KeyboardInterrupt — stopping workers.")
        integrand.end()
    finally:
        integrand.end()


def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python state_import.py <settings_file> <state_file>")
        return

    print("Starting state import test...")

    run_state_import(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
