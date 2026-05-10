
def madnis_state_to_gammaboard_api(state_file: str, output: str):
    import pickle
    from glnis.scripts.sampler_comparison import SamplerCompData, MADNIS_KEY
    from glnis.utils.helpers import verify_path
    from pathlib import Path

    state_file = verify_path(state_file)
    with state_file.open("rb") as f:
        data: SamplerCompData = pickle.load(f)
    madnis_state = data.integrator_states.get(MADNIS_KEY)
    if madnis_state is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with Path(output).open("wb") as f:
            pickle.dump(dict(
                madnis_blob=madnis_state.madnis_blob,
                torch_cpu_rng_state=madnis_state.torch_cpu_rng_state,
                torch_gpu_rng_state=madnis_state.torch_gpu_rng_state,
            ), f)
