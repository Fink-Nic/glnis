from glnis.utils.helpers import verify_path

verify_path("glnis/settings", 1)
verify_path("glnis/settings.hi", 1, suffix=".toml")
