# Installation
* A nix flake is provided, if necessary
* Install in editable mode
```bash
pip install -e .
```
* GammaLoop installation from:
```
https://github.com/alphal00p/gammaloop
```
Follow the instructions in order to install the python API (make sure you install it in the same venv as `glnis`).

# Setup

In order to easily be able to use the examples provided, please move the `glnis_gammaloop_examples` folder inside of your gammaloop installation folder. Then run `glnis setdef -d <path/to/glnis_gammaloop_examples>` to tell the script where to find them. 

Do not change the name of `glnis_gammaloop_examples`, the example runcards rely on it.

## Creating your gammaloop states
All the examples can be generated and integrated using the following shell command:
```bash
./gammaloop glnis_gammaloop_examples/<path/to/example_runcard.toml> run generate integrate -c "quit -o"
```

## Testing your setup
Run a single training step and integrate a batch of samples:
```bash
glnis stest -f <path/to/example.toml>
```
When encountering `dlopen` errors, you may need to execute the script from within your `gammaloop` installation folder.

## Minimal working training run, including output and saving of data
```bash
glnis tprog -f <path/to/example.toml>
```

# Other scripts and configs
* Run `glnis --help` for a summary of available scripts. 
* Run `glnis <subcommand> --help` for a list of options for a given script.
* A comprehensive list of options for the samplers, integrands, parameterisation chain, scripts and plotting can be found in the default settings file at `settings/default.toml`.
