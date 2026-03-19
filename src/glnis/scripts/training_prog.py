# type: ignore

def run_training_prog(file: str,
                      comment: str = "",
                      no_output: bool = False,
                      no_plot: bool = False,
                      only_plot: bool = False,
                      export_states: bool = True,
                      ) -> None:
    from glnis.scripts.sampler_comparison import run_sampler_comp

    run_sampler_comp(file=file,
                     comment=comment,
                     no_naive=True,
                     no_vegas=True,
                     no_havana=True,
                     no_output=no_output,
                     no_plot=no_plot,
                     only_plot=only_plot,
                     export_states=export_states,
                     subroutine="training_prog",)
