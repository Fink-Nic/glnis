# type: ignore

from pathlib import Path
from glnis.utils.helpers import verify_path
import tomlkit


def run_set_default_gl_path(path: str) -> None:
    try:
        default_settings_file = verify_path("settings/default.toml")
        path: Path = Path(path)
        if not path.is_absolute():
            path = path.resolve()
        if not path.is_dir():
            raise ValueError(f"Provided path '{path}' is not a directory.")

        magic_file = path / ".glnis.finder"
        if not magic_file.is_file():
            raise ValueError(
                f"Provided directory '{path}' does not contain '.glnis.finder'. Please ensure you have provided the correct path to the GammaLoop examples.")
        path = path
        with open(magic_file, "r") as f:
            # Check for magic number
            first_line = f.readline().strip()
            if not first_line == "658123":
                raise ValueError(
                    f"Invalid magic number in '{magic_file}'. Expected '658123', got '{first_line}'.")

        # Update the default settings file
        with open(default_settings_file, "rb") as f:
            settings = tomlkit.load(f)
        settings['gammaloop']['state_dir'] = str(path)
        with open(default_settings_file, "w") as f:
            tomlkit.dump(settings, f)
        print(f"Successfully set path to GammaLoop examples to '{path}'.")

    except Exception as e:
        from traceback import format_exc
        print(format_exc())
        print(f"Error: {e}")
        return
