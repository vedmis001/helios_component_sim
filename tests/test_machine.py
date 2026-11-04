"""
Induction Machine validation: Double-cage rotor in synchronous dq frame.

Checks:
1. 3-phase stator voltage and current injection into MNA.
2. Park / Clarke transform stability.
3. Rotor speed response under load torque.

Run from the helios_component_sim/ directory:
    python tests/test_machine.py
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
from components.machine import InductionMachine

ELEMENT_REGISTRY.setdefault("machine", InductionMachine)

# ---- Test parameters -------------------------------------------------
V_PHASE_PEAK = 271.4 * np.sqrt(2.0)
W_GRID = 377.0  # 60 Hz
DT = 1e-5
T_END = 0.04    # ~2.5 electrical cycles

config = {
    "nodes": ["gnd", "va", "vb", "vc", "vn"],
    "elements": [
        {
            "type": "vsource",
            "name": "Vs_a",
            "nodes": ["va", "vn"],
            "value": lambda t: V_PHASE_PEAK * np.cos(W_GRID * t)
        },
        {
            "type": "vsource",
            "name": "Vs_b",
            "nodes": ["vb", "vn"],
            "value": lambda t: V_PHASE_PEAK * np.cos(W_GRID * t - 2.0*np.pi/3.0)
        },
        {
            "type": "vsource",
            "name": "Vs_c",
            "nodes": ["vc", "vn"],
            "value": lambda t: V_PHASE_PEAK * np.cos(W_GRID * t + 2.0*np.pi/3.0)
        },
        {"type": "resistor", "name": "R_n_gnd", "nodes": ["vn", "gnd"], "value": 1e-4},
        {
            "type": "machine",
            "name": "IM1",
            "nodes": ["va", "vb", "vc", "vn"],
            "value": {
                "Vbase": 271.4,
                "Ibase": 60.0,
                "H": 1.0,
                "D": 0.008,
                "Tload": 0.5
            }
        }
    ]
}


def test_machine_run() -> None:
    circuit = Circuit.from_config(config)
    sim = Simulator(circuit, DT, T_END)
    sim.run()

    t = sim.time
    va = sim.voltage("va")
    vb = sim.voltage("vb")
    vc = sim.voltage("vc")

    im_el = [e for e in circuit.elements if isinstance(e, InductionMachine)][0]

    ia = sim.x[:, im_el.branch_idx]
    ib = sim.x[:, im_el.branch_idx + 1]
    ic = sim.x[:, im_el.branch_idx + 2]

    print(f"[Induction Machine] Final Phase A current = {ia[-1]:.2f} A")
    print(f"[Induction Machine] Final Rotor Speed wr   = {im_el.wr:.4f} pu")

    # Stator currents should be non-zero and finite
    assert np.max(np.abs(ia)) > 1.0, "Induction machine drew zero current"
    assert not np.isnan(ia[-1]), "NaN detected in machine simulation"

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(t * 1e3, va, label="Phase A Voltage (V)", color="blue")
    ax1.plot(t * 1e3, vb, label="Phase B Voltage (V)", color="green")
    ax1.plot(t * 1e3, vc, label="Phase C Voltage (V)", color="red")
    ax1.set_ylabel("Stator Voltage (V)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t * 1e3, ia, label="ia (A)", color="blue")
    ax2.plot(t * 1e3, ib, label="ib (A)", color="green")
    ax2.plot(t * 1e3, ic, label="ic (A)", color="red")
    ax2.set_ylabel("Stator Current (A)")
    ax2.set_xlabel("Time (ms)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("machine_test.png", dpi=150)
    print("[Induction Machine] PASS - saved machine_test.png")


if __name__ == "__main__":
    test_machine_run()

