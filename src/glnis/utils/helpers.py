# type: ignore
import math
from enum import StrEnum
from typing import Dict, Sequence, List, Any, Iterable


class Colour(StrEnum):
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def error_fmter(value: float, error: float, prec_error: int = 2) -> str:
    """
    Format a value and its error in scientific notation with a given number of significant digits for the error.

    Examples:
        value = 1234.5678, error = 111.11, prec_error = 2 -> "1.23(11)e+04"
        value = 12.345, error = 111.111, prec_error = 1 -> "1.2(11.1)e+01"
        value = 0.0123, error = 0.001234, prec_error = 3 -> "1.230(123)e-02"
    """
    if error < 0:
        raise ValueError("Error must be positive.")
    prec_error = max(1, prec_error)

    if value == 0:
        log10val = 0
    else:
        log10val = math.floor(math.log10(abs(value)))

    exp10val = 10.0**log10val

    # Normalize both value and error to the same order of magnitude
    val_norm = value / exp10val
    err_norm = error / exp10val

    # Set prec: the significant number of digits such that prec_error number
    # of significant digits are shown for the error
    if error == 0:
        val_str = f"{val_norm:.{prec_error}f}"
        return f"{val_str}(0)e{log10val:+03d}"

    log10err_norm = math.floor(math.log10(err_norm))

    if log10err_norm >= 0:
        prec = prec_error
    else:
        prec = prec_error - log10err_norm - 1

    # Get digits without scientific notation
    val_str = f"{val_norm:.{prec}f}"
    if log10err_norm >= 0:
        err_str = f"{err_norm:.{prec}f}"
    else:
        err_str = f"{err_norm:.{prec}e}".replace(".", "")[:prec_error]
    # I don't think this can happen since error>0, but if the error is somehow rounded
    # down to zero, err_str will be empty and we default to
    if not err_str:
        err_str = '0' * prec_error

    return f"{val_str}({err_str})e{log10val:+03d}"


def chunks(ary: Sequence, n_chunks: int) -> Iterable[Sequence]:
    """
    Like numpy.array_split, but works for all sequences
    """
    ln_ary = len(ary)
    if n_chunks > ln_ary or n_chunks < 1:
        raise ValueError(
            "the number of chunks should be at least 1, and at most len(ary)")
    n_long = ln_ary % n_chunks
    ln_long = ln_ary // n_chunks + 1
    total_long = n_long*ln_long
    ln_short = ln_ary // n_chunks

    for start in range(0, total_long, ln_long):
        yield ary[start:start+ln_long]
    for start in range(total_long, ln_ary, ln_short):
        yield ary[start:start+ln_short]


def overwrite_settings(orig_dict: Dict[str, Any], new_dict: Dict[str, Any],
                       always_overwrite: List[str] = [],
                       ) -> Dict[str, Any]:
    """
    Used to Overwrite the default settings with the specified settings file.
    """
    if "overwrite" in new_dict:
        orig_dict["overwrite"] = new_dict["overwrite"]
        overwrite = new_dict["overwrite"]
        for joined_keys, value in overwrite.items():
            keys = joined_keys.split(".")
            d = orig_dict
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value

    for force in always_overwrite:
        if force in new_dict.keys():
            orig_dict[force] = new_dict[force]

    for key, val in new_dict.items():
        if isinstance(val, Dict):
            tmp = overwrite_settings(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]

    return orig_dict
