# Installation
* Install in editable mode
```bash
pip install -e .
```
* Gammaloop `hedge_numerator` branch installation from:
```
https://github.com/alphal00p/gammaloop
```
Follow the instructions in order to install the python API.

# Workflow
## Settings files
Inside of `settings/`, there are a number of examples. A list and explanation of the parameters can be found within `settings/default.toml`. It is recommended to follow the naming convention of the examples, this also applies to the names of the `GammaLoop` runcards, `.dot` files and state folders.

## Setup
* In order to easily be able to use the examples provided, please point the `settings/default.toml` to the folder containing your 'GammaLoop' states:
```toml
[gammaloop_state]
state_folder = "/path/to/your/folder"
```

* Place the contents of `gammaloop_files/` inside your `GammaLoop` folder
These are the runcards and `.dot` files required to generate the example states.

## Setting up your gammaloop states
You can use the makefile provided inside of `gammaloop_files` in order to:
* Generate a state from one of the provided runcards inside of `gammaloop_files/runcards`. 
Example (to be run from inside your `GammaLoop` folder):
```bash
make -f gl.makefile generate NAME=scalar_box
```
* Integrate an example:
```bash
make -f gl.makefile integrate NAME=scalar_box
```

## Testing the compatibility with glnis
Example:
```bash
glnis stest -s settings/scalar_box.toml
```

## Minimal working training run, including output of data
Example:
```bash
glnis tprog -s settings/scalar_box.toml
```

## Quirks
Follow the example in `scripts/training_prog.py`:
* In order to properly catch `KeyboardInterrupt` exceptions, set the `SIGINT` signal handling via
```python
signal.signal(signal.SIGINT, signal.default_int_handler)
```
* Once you are done (or want to prematurely exit), call `end()` on your `MPIntegrand` objects:
```python
integrand.end()
```