"""
Third validation: underdamped series RLC driven by a DC voltage step.

    Vs --- R --- n2 --- L --- n3 --- C --- gnd

Output: capacitor voltage v(n3).

Closed form (underdamped, alpha < w0):
    alpha = R / (2L),  w0 = 1 / sqrt(LC),  wd = sqrt(w0^2 - alpha^2)
    Vc(t) = Vs * (1 - exp(-alpha t) * (cos(wd t) + (alpha / wd) * sin(wd t)))

A second, independent reference comes from scipy's solve_ivp on the state
equations di/dt = (Vs - R i - vc) / L,  dvc/dt = i / C.

Run from the helios_sim/ directory:
    python tests/test_rlc.py
    python tests/test_rlc.py --convergence    # also halves dt repeatedly (slower)
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no GUI; figure is saved to a file
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.circuit import Circuit
from core.simulator import Simulator

# ---- Test parameters -------------------------------------------------
R = 5.0         # ohms
L = 10e-3       # henries
C = 100e-6      # farads
VS = 1.0        # volts
DT = 1e-6       # seconds
T_END = 20e-3   # seconds (about five decay time constants, 1/alpha = 4 ms)

RMSE_TOL = 1e-3  # volts

ALPHA = R / (2.0 * L)
W0 = 1.0 / np.sqrt(L * C)
assert ALPHA < W0, "closed form below is only valid for the underdamped case"
WD = np.sqrt(W0**2 - ALPHA**2)


def build_config() -> dict:
    return {
        "nodes": ["gnd", "n1", "n2", "n3"],
        "elements": [
            {"type": "vsource", "name": "Vs", "nodes": ["n1", "gnd"], "value": VS},
            {"type": "resistor", "name": "R1", "nodes": ["n1", "n2"], "value": R},
            {"type": "inductor", "name": "L1", "nodes": ["n2", "n3"], "value": L},
            {"type": "capacitor", "name": "C1", "nodes": ["n3", "gnd"], "value": C},
        ],
    }


def run_sim(dt: float) -> Simulator:
    sim = Simulator(Circuit.from_config(build_config()), dt, T_END)
    sim.run()
    return sim


def analytical(t: np.ndarray) -> np.ndarray:
    return VS * (1.0 - np.exp(-ALPHA * t) * (np.cos(WD * t) + (ALPHA / WD) * np.sin(WD * t)))


def ivp_reference(t: np.ndarray) -> np.ndarray:
    def rhs(_, x):
        i, vc = x
        return [(VS - R * i - vc) / L, i / C]

    sol = solve_ivp(rhs, [t[0], t[-1]], [0.0, 0.0], t_eval=t, rtol=1e-10, atol=1e-12)
    return sol.y[1]


def rmse_of(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def convergence_study() -> None:
    """Halve dt and watch how the error shrinks."""
    print("\nconvergence study (RMSE vs analytical):")
    prev = None
    for dt in [2e-6, 1e-6, 5e-7, 2.5e-7]:
        sim = run_sim(dt)
        r = rmse_of(sim.voltage("n3"), analytical(sim.time))
        ratio = f"   ratio to previous: {prev / r:.2f}" if prev else ""
        print(f"  dt = {dt:.2e}  RMSE = {r:.3e} V{ratio}")
        prev = r


def main() -> None:
    sim = run_sim(DT)

    t = sim.time
    vc_sim = sim.voltage("n3")
    vc_exact = analytical(t)
    vc_ivp = ivp_reference(t)

    err = vc_sim - vc_exact
    rmse = rmse_of(vc_sim, vc_exact)
    max_err = float(np.max(np.abs(err)))
    ref_gap = float(np.max(np.abs(vc_exact - vc_ivp)))

    print(f"alpha = {ALPHA:.1f} 1/s, w0 = {W0:.1f} rad/s, wd = {WD:.1f} rad/s")
    print(f"simulator vs analytical: RMSE = {rmse:.3e} V, max |e| = {max_err:.3e} V")
    print(f"analytical vs solve_ivp: max gap = {ref_gap:.3e} V (should be tiny)")
    print(f"peak Vc: simulator {vc_sim.max():.4f} V, analytical {vc_exact.max():.4f} V")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(t, vc_exact, label="analytical")
    ax1.plot(t, vc_sim, "--", label="simulator")
    ax1.set_ylabel("Vc (V)")
    ax1.legend()
    ax2.plot(t, err)
    ax2.set_ylabel("error (V)")
    ax2.set_xlabel("time (s)")
    fig.savefig("rlc_test.png", dpi=150)
    print("saved rlc_test.png")

    assert ref_gap < 1e-6, "the two reference solutions disagree; check the test itself"
    assert rmse < RMSE_TOL, f"RMSE {rmse:.3e} exceeds tolerance {RMSE_TOL:.1e}"
    print("PASS")

    if "--convergence" in sys.argv:
        convergence_study()


if __name__ == "__main__":
    main()