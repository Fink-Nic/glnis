# type: ignore
from glnis.scripts.sampler_comparison import SamplerCompData, run_sampler_comp


def run_training_prog(file: str,
                      comment: str = "",
                      no_output: bool = False,
                      no_plot: bool = False,
                      export_states: bool = True,
                      ) -> SamplerCompData | None:

    return run_sampler_comp(file=file,
                            comment=comment,
                            no_naive=True,
                            no_vegas=True,
                            no_havana=True,
                            no_output=no_output,
                            no_plot=no_plot,
                            export_states=export_states,
                            subroutine="training_prog",)
