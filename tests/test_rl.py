"""
Second validation: series RL driven by a DC voltage step.

    Vs --- R --- n2 --- L --- gnd

Analytical solution:  i(t) = (Vs / R) * (1 - exp(-R * t / L))

The series current is read from the voltage drop across the resistor,
i = (v(n1) - v(n2)) / R, which doesn't depend on how your voltage-source
branch current is signed.

Run from the helios_sim/ directory:
    python tests/test_rl.py
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
R = 10.0        # ohms
L = 10e-3       # henries  (L/R = 1 ms)
VS = 1.0        # volts
DT = 1e-6       # seconds
T_END = 5e-3    # seconds, five time constants

I_FINAL = VS / R
REL_RMSE_TOL = 1e-3   # RMSE as a fraction of the final current

# ---- Circuit ---------------------------------------------------------
config = {
    "nodes": ["gnd", "n1", "n2"],
    "elements": [
        {"type": "vsource", "name": "Vs", "nodes": ["n1", "gnd"], "value": VS},
        {"type": "resistor", "name": "R1", "nodes": ["n1", "n2"], "value": R},
        {"type": "inductor", "name": "L1", "nodes": ["n2", "gnd"], "value": L},
    ],
}


def main() -> None:
    circuit = Circuit.from_config(config)
    sim = Simulator(circuit, DT, T_END)
    sim.run()

    t = sim.time
    i_sim = (sim.voltage("n1") - sim.voltage("n2")) / R
    i_exact = I_FINAL * (1.0 - np.exp(-R * t / L))

    err = i_sim - i_exact
    rmse = float(np.sqrt(np.mean(err**2)))
    max_err = float(np.max(np.abs(err)))

    print(f"RMSE    = {rmse:.3e} A  ({rmse / I_FINAL:.2e} of final current)")
    print(f"max |e| = {max_err:.3e} A")

    # Informational only: how your voltage-source branch current compares.
    # Depending on your stamp convention it equals +i or -i, so this is not
    # asserted. Use it to confirm you know which sign your convention gives.
    i_branch = sim.branch_current("Vs")
    print(f"branch current vs series current: "
          f"max|ib - i| = {np.max(np.abs(i_branch - i_sim)):.3e}, "
          f"max|ib + i| = {np.max(np.abs(i_branch + i_sim)):.3e}")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(t, i_exact, label="analytical")
    ax1.plot(t, i_sim, "--", label="simulator")
    ax1.set_ylabel("i (A)")
    ax1.legend()
    ax2.plot(t, err)
    ax2.set_ylabel("error (A)")
    ax2.set_xlabel("time (s)")
    fig.savefig("rl_test.png", dpi=150)
    print("saved rl_test.png")

    assert rmse / I_FINAL < REL_RMSE_TOL, (
        f"relative RMSE {rmse / I_FINAL:.3e} exceeds tolerance {REL_RMSE_TOL:.1e}"
    )
    print("PASS")


if __name__ == "__main__":
    main()