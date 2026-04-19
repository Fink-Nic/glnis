# type: ignore

def run_state_import(
    file: str,
) -> None:
    from glnis.utils.helpers import shell_print, verify_path
    from glnis.scripts.sampler_comparison import SamplerCompData
    from glnis.core.accumulator import DefaultAccumulator

    import signal
    from torch import load

    from glnis.core.integrator import Integrator
    from glnis.core.parser import SettingsParser

    file = verify_path(file)
    with file.open('rb') as f:
        SData: SamplerCompData = load(f, weights_only=False)
    if not isinstance(SData, SamplerCompData):
        shell_print(f"File {file} does not contain SamplerCompData. Exiting.")
        return

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        shell_print(f"Working on file {file}")
        Settings = SettingsParser(SData.settings)

        Settings.settings["layered_integrator"]["integrator_type"] = "madnis"
        madnis_integrator = Integrator.from_settings(
            Settings.settings
        )
        integrand = madnis_integrator.integrand

        shell_print(f"MadNIS is using device: {madnis_integrator.madnis.dummy.device}")
        shell_print(f"MadNIS is using scheduler: {madnis_integrator.madnis.scheduler}")
        shell_print(f"Integrand discrete dims: {integrand.discrete_dims}")

        # Parse GammaLoop results
        # Creating and importing all the samplers
        for name, state in SData.integrator_states.items():
            shell_print(f"Found {name} state")
            obs = SData.result.get(name, None)
            if obs is None:
                shell_print(f"Found no observables for {name}. Skipping...")
                continue

            try:
                integrator = Integrator.from_state(state, integrand)
            except Exception as e:
                shell_print(f"Failed at 'from_state' for {name} with error: {e}. Skipping...")
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
                shell_print(f"type of integrator: {type(integrator)}")
                shell_print(f"Successfully imported {name} state")
                acc: DefaultAccumulator = integrator.integrate(n_points, progress_report=False)
                obs = acc.statistics.result
                shell_print(f"Result for {name} after importing, run {i + 1}:")
                if obs.real_error:
                    shell_print(
                        f"    RE : {obs.real_central_value:.8e} +- {obs.real_error:.8e}, RSD = {obs.real_rsd:.3f}\n")
                if obs.imag_error:
                    shell_print(
                        f"    IM : {obs.imag_central_value:.8e} +- {obs.imag_error:.8e}, RSD = {obs.imag_rsd:.3f}\n")
                try:
                    integrator.import_state(state)
                except Exception as e:
                    shell_print(f"Failed at 'import_state' for {name} with error: {e}. Skipping...")
                    continue

    except KeyboardInterrupt:
        shell_print("\nCaught KeyboardInterrupt — stopping workers.")
        integrator.free()
    finally:
        integrator.free()


def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python state_import.py <state_file>")
        return

    print("Starting state import test...")

    run_state_import(sys.argv[1])


if __name__ == "__main__":
    main()
