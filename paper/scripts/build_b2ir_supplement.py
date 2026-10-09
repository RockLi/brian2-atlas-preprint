"""Compatibility entry point for the AtlasIR supplementary-figure builder."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("build_atlasir_supplement.py")), run_name="__main__")
