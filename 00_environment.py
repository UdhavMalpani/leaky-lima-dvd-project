"""
Step 0: record the software environment (needed for the report's "Setup"
section) and check that ngspice can be found and actually runs.

Output: results/env.txt
"""
import platform
import subprocess
import sys
from importlib import metadata

import config

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "matplotlib", "joblib"]


def ngspice_version():
    try:
        out = subprocess.run([config.NGSPICE, "-v"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        sys.exit(f"ngspice not runnable at {config.NGSPICE!r}: {e}\n"
                 "Install ngspice or set the NGSPICE environment variable.")
    for line in (out.stdout + out.stderr).splitlines():
        if "ngspice-" in line:
            return line.strip("* ").strip()
    return "unknown"


def main():
    lines = [
        f"technology      : {config.TECH}  (model file {config.MODEL_FILE.name})",
        f"nominal VDD     : {config.VDD_NOM} V",
        f"test circuit    : {config.TEST_CIRCUIT}",
        f"simulator       : {ngspice_version()}",
        f"ngspice binary  : {config.NGSPICE}",
        f"python          : {platform.python_version()}",
        f"platform        : {platform.platform()}",
        f"cpu cores used  : {config.N_WORKERS}",
    ]
    for p in PACKAGES:
        try:
            lines.append(f"{p:16s}: {metadata.version(p)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{p:16s}: NOT INSTALLED")
    text = "\n".join(lines) + "\n"
    (config.RESULTS_DIR / "env.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
