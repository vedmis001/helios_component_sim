"""
First validation: series RC driven by a DC voltage step.

    Vs --- R --- n2 --- C --- gnd

Analytical solution:  Vc(t) = Vs * (1 - exp(-t / (R*C)))

Run from the helios_sim/ directory:
    python tests/test_rc.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no GUI; figure is saved to a file
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.circuit import Circuit
from core.simulator import Simulator

# ---- Test parameters -------------------------------------------------
R = 100.0       # ohms
C = 1e-4        # farads
VS = 1.0        # volts
DT = 1e-6       # seconds
T_END = 0.05    # seconds, five time constants (RC = 0.01 s)

RMSE_TOL = 1e-4  # volts; tighten this once everything works

# ---- Circuit ---------------------------------------------------------
# VoltageSource nodes are [positive, negative], so the source is
# ["n1", "gnd"], which forces v(n1) - v(gnd) = +VS.
config = {
    "nodes": ["gnd", "n1", "n2"],
    "elements": [
        {"type": "vsource", "name": "Vs", "nodes": ["n1", "gnd"], "value": VS},
        {"type": "resistor", "name": "R1", "nodes": ["n1", "n2"], "value": R},
        {"type": "capacitor", "name": "C1", "nodes": ["n2", "gnd"], "value": C},
    ],
}


def main() -> None:
    circuit = Circuit.from_config(config)
    sim = Simulator(circuit, DT, T_END)
    sim.run()

    t = sim.time
    v_sim = sim.voltage("n2")
    v_exact = VS * (1.0 - np.exp(-t / (R * C)))

    err = v_sim - v_exact
    rmse = float(np.sqrt(np.mean(err**2)))
    max_err = float(np.max(np.abs(err)))

    print(f"RMSE    = {rmse:.3e} V")
    print(f"max |e| = {max_err:.3e} V")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(t, v_exact, label="analytical")
    ax1.plot(t, v_sim, "--", label="simulator")
    ax1.set_ylabel("Vc (V)")
    ax1.legend()
    ax2.plot(t, err)
    ax2.set_ylabel("error (V)")
    ax2.set_xlabel("time (s)")
    fig.savefig("rc_test.png", dpi=150)
    print("saved rc_test.png")

    assert rmse < RMSE_TOL, f"RMSE {rmse:.3e} exceeds tolerance {RMSE_TOL:.1e}"
    print("PASS")


if __name__ == "__main__":
    main()