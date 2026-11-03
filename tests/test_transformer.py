"""
Transformer validation, using the HELIOS 138/35 kV, 150 MVA transformer
(X = 0.15 pu, R = 0.001 pu) as a single-phase per-phase equivalent.

    Vs --- [ transformer 138:35, leakage on primary ] --- R_load (1.0 pu)

Two checks:

1. DC step on the primary (dynamic check of the leakage inductance).
   With R_ref = a^2 * R_load, tau = L / (R + R_ref):
       i1(t)  = Vs / (R + R_ref) * (1 - exp(-t / tau))
       vs(t)  = a * R_load * i1(t)

2. AC steady state (checks ratio, impedance referral and phase shift).
   Phasor result, with Zl = R + jwL:
       Vs / Vp = (1/a) * (a^2 R_load) / (Zl + a^2 R_load)
   Also checks power balance: P_in = P_load + P_R(leakage).

Run from the helios_sim/ directory:
    python tests/test_transformer.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # no GUI; figures are saved to files
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.circuit import Circuit, ELEMENT_REGISTRY
from core.simulator import Simulator
from components.transformer import Transformer, leakage_from_pu

# Register the transformer so Circuit.from_config can build it.
ELEMENT_REGISTRY.setdefault("transformer", Transformer)

# ---- Transformer and load -------------------------------------------
V_LL_P = 138e3          # primary line-to-line, V
V_LL_S = 35e3           # secondary line-to-line, V
S_3PH = 150e6           # VA
X_PU = 0.15
R_PU = 0.001
F = 60.0
W = 2.0 * np.pi * F

V_PHASE = V_LL_P / np.sqrt(3.0)                 # per-phase source voltage
RATIO = V_LL_P / V_LL_S
R_LEAK, L_LEAK = leakage_from_pu(X_PU, R_PU, V_LL_P, S_3PH, F)
R_LOAD = V_LL_S**2 / S_3PH                      # 1.0 pu resistive load, secondary side
R_REF = RATIO**2 * R_LOAD                       # load referred to the primary
TAU = L_LEAK / (R_LEAK + R_REF)

# ---- Tolerances ------------------------------------------------------
DC_REL_RMSE_TOL = 2e-3      # DC step has the t = 0 startup artifact (first-order in dt)
AC_REL_RMSE_TOL = 1e-5      # steady state has no discontinuity, so this can be tight
POWER_REL_TOL = 1e-6


def build_config(source_value) -> dict:
    return {
        "nodes": ["gnd", "n1", "n2"],
        "elements": [
            {"type": "vsource", "name": "Vs", "nodes": ["n1", "gnd"], "value": source_value},
            {"type": "transformer", "name": "T1", "nodes": ["n1", "gnd", "n2", "gnd"],
             "value": {"ratio": RATIO, "R": R_LEAK, "L": L_LEAK}},
            {"type": "resistor", "name": "Rload", "nodes": ["n2", "gnd"], "value": R_LOAD},
        ],
    }


def run_sim(source_value, dt: float, t_end: float) -> Simulator:
    sim = Simulator(Circuit.from_config(build_config(source_value)), dt, t_end)
    sim.run()
    return sim


def test_dc_step() -> None:
    dt, t_end = 1e-6, 5 * TAU
    sim = run_sim(V_PHASE, dt, t_end)

    t = sim.time
    i1 = V_PHASE / (R_LEAK + R_REF) * (1.0 - np.exp(-t / TAU))
    v_exact = RATIO * R_LOAD * i1
    v_sim = sim.voltage("n2")

    err = v_sim - v_exact
    rel_rmse = float(np.sqrt(np.mean(err**2)) / v_exact[-1])
    print(f"[DC step] tau = {TAU * 1e3:.3f} ms, final vs = {v_exact[-1]:.2f} V")
    print(f"[DC step] relative RMSE = {rel_rmse:.3e}, "
          f"relative max error = {np.max(np.abs(err)) / v_exact[-1]:.3e}")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(t * 1e3, v_exact, label="analytical")
    ax1.plot(t * 1e3, v_sim, "--", label="simulator")
    ax1.set_ylabel("secondary voltage (V)")
    ax1.legend()
    ax2.plot(t * 1e3, err)
    ax2.set_ylabel("error (V)")
    ax2.set_xlabel("time (ms)")
    fig.savefig("transformer_dc_step.png", dpi=150)

    assert rel_rmse < DC_REL_RMSE_TOL, (
        f"DC step relative RMSE {rel_rmse:.3e} exceeds {DC_REL_RMSE_TOL:.1e}")
    print("[DC step] PASS")


def test_ac_steady_state() -> None:
    steps_per_cycle = 1600
    dt = 1.0 / (F * steps_per_cycle)    # an exact number of steps per cycle
    n_cycles = 6                        # the transient (tau = 0.4 ms) is long gone by then
    t_end = n_cycles / F
    v_peak = V_PHASE * np.sqrt(2.0)

    sim = run_sim(lambda t: v_peak * np.sin(W * t), dt, t_end)

    # Analytical steady state
    z_leak = R_LEAK + 1j * W * L_LEAK
    gain = (1.0 / RATIO) * R_REF / (z_leak + R_REF)
    v_exact = v_peak * abs(gain) * np.sin(W * sim.time + np.angle(gain))

    win = slice(-2 * steps_per_cycle, None)     # last two full cycles
    t = sim.time[win]
    v1 = sim.voltage("n1")[win]
    v2 = sim.voltage("n2")[win]
    err = v2 - v_exact[win]
    rel_rmse = float(np.sqrt(np.mean(err**2)) / (v_peak * abs(gain)))

    # Fit gain and phase from the simulated secondary voltage
    A = np.column_stack([np.sin(W * t), np.cos(W * t)])
    c = np.linalg.lstsq(A, v2, rcond=None)[0]
    gain_fit = np.hypot(*c) / v_peak
    phase_fit = np.degrees(np.arctan2(c[1], c[0]))

    # Power balance over an integer number of cycles. The source current's
    # sign depends on your stamp convention, so use the magnitude.
    i_src = sim.branch_current("Vs")[win]
    p_in = abs(np.mean(v1 * i_src))
    p_load = np.mean(v2**2) / R_LOAD
    p_leak = R_LEAK * np.mean(i_src**2)
    p_rel = abs(p_in - (p_load + p_leak)) / p_in

    print(f"[AC] gain  simulated {gain_fit:.6f}   expected {abs(gain):.6f}")
    print(f"[AC] phase simulated {phase_fit:.4f} deg   expected {np.degrees(np.angle(gain)):.4f} deg")
    print(f"[AC] waveform relative RMSE = {rel_rmse:.3e}")
    print(f"[AC] P_in = {p_in:.4e} W, P_load + P_leak = {p_load + p_leak:.4e} W, "
          f"relative difference = {p_rel:.3e}")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(t * 1e3, v_exact[win], label="analytical")
    ax1.plot(t * 1e3, v2, "--", label="simulator")
    ax1.set_ylabel("secondary voltage (V)")
    ax1.legend()
    ax2.plot(t * 1e3, err)
    ax2.set_ylabel("error (V)")
    ax2.set_xlabel("time (ms)")
    fig.savefig("transformer_ac.png", dpi=150)

    assert rel_rmse < AC_REL_RMSE_TOL, (
        f"AC relative RMSE {rel_rmse:.3e} exceeds {AC_REL_RMSE_TOL:.1e}")
    assert p_rel < POWER_REL_TOL, (
        f"power balance off by {p_rel:.3e}, tolerance {POWER_REL_TOL:.1e}")
    print("[AC] PASS")


if __name__ == "__main__":
    test_dc_step()
    test_ac_steady_state()
    print("saved transformer_dc_step.png, transformer_ac.png")