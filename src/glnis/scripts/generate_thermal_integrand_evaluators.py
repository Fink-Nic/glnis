# type:ignore

def run_generate_thermal_integrand_evaluators(force_rebuild: bool = False) -> None:
    from time import perf_counter
    from glnis.utils.helpers import shell_print, verify_path
    try:
        import kaapos.integrands as kintegrands
    except:
        raise ImportError("Failed to import thermal integrand module.")

    params = [6.283185307179586, 3.141592653589793, 1.0]
    examples = dict(
        sunrise="thermal_integrand_examples/sunset",
        mercedes="thermal_integrand_examples/mercedes",
        bugblatter="thermal_integrand_examples/bugblatterPQ",
    )

    for example, path_to_example in examples.items():
        p = verify_path(path_to_example)
        before = perf_counter()
        shell_print(f"Generating example '{example}': {p}")
        kintegrands.SymbolicaIntegrand(params=params, path_to_example=p, force_rebuild=force_rebuild)
        after = perf_counter()
        shell_print(f"Finished generating example '{example}' in {after - before:.2f} seconds.")
