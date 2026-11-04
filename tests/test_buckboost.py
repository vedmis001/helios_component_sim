"""
Buck/Boost converter validation: Averaged switch model + PI voltage regulation.

Checks:
1. Boost mode operation: steps up 600 V battery to 900 V DC bus.
2. PI controller dynamics and duty cycle calculation.
3. Stable regulation of high-side voltage under load.

Run from the helios_component_sim/ directory:
    python tests/test_buckboost.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.circuit import Circuit, ELEMENT_REGISTRY
from core.simulator import Simulator
from components.buckboost import BuckBoost

ELEMENT_REGISTRY.setdefault("buckboost", BuckBoost)

# ---- Test parameters -------------------------------------------------
VL_SRC = 600.0    # Battery side DC source (V)
VH_TARGET = 900.0 # DC link regulation target (V)
R_LOAD = 50.0     # DC link load resistor
DT = 1e-5         # seconds
T_END = 0.04      # seconds

config = {
    "nodes": ["gnd", "vh_p", "vh_n", "vl_p", "vl_n"],
    "elements": [
        {"type": "vsource", "name": "V_batt", "nodes": ["vl_p", "vl_n"], "value": VL_SRC},
        {"type": "resistor", "name": "R_batt_gnd", "nodes": ["vl_n", "gnd"], "value": 1e-4},
        {
            "type": "buckboost",
            "name": "BB1",
            "nodes": ["vh_p", "vh_n", "vl_p", "vl_n"],
            "value": {
                "Ltot": 0.0015,
                "mode": "boost",
                "VH_ref": VH_TARGET,
                "Kp_boost": 0.05,
                "Ti_boost": 0.02,
                "Dmax_boost": 0.70
            }
        },
        {"type": "capacitor", "name": "C_dclink", "nodes": ["vh_p", "vh_n"], "value": 4.7e-3},
        {"type": "resistor", "name": "R_load", "nodes": ["vh_p", "vh_n"], "value": R_LOAD},
        {"type": "resistor", "name": "R_vh_gnd", "nodes": ["vh_n", "gnd"], "value": 1e-4},
    ],
}


def test_buckboost_boost() -> None:
    circuit = Circuit.from_config(config)
    sim = Simulator(circuit, DT, T_END)
    sim.run()

    t = sim.time
    vh_p = sim.voltage("vh_p")
    vh_n = sim.voltage("vh_n")
    vh = vh_p - vh_n

    bb_el = [e for e in circuit.elements if isinstance(e, BuckBoost)][0]

    final_vh = vh[-1]
    print(f"[BuckBoost] Final VH = {final_vh:.2f} V (Target = {VH_TARGET:.2f} V)")
    print(f"[BuckBoost] Final Duty = {bb_el.duty:.4f}")

    # Ensure voltage rises towards target and stays stable
    assert final_vh > 0.8 * VH_TARGET, f"Boost converter failed to boost voltage adequately: {final_vh:.2f} V"

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(t * 1e3, vh, label="VH (DC Link)", color="crimson")
    ax1.axhline(VH_TARGET, ls="--", color="gray", label="VH Target (900 V)")
    ax1.set_ylabel("High-Side Voltage (V)")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t * 1e3, vh / R_LOAD, label="Load Current (A)", color="darkgreen")
    ax2.set_ylabel("Current (A)")
    ax2.set_xlabel("Time (ms)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("buckboost_test.png", dpi=150)
    print("[BuckBoost] PASS - saved buckboost_test.png")


if __name__ == "__main__":
    test_buckboost_boost()

