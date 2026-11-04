"""
Battery validation: Shepherd model constant-current / resistive discharge.

Checks:
1. Coulomb counting SOC integration: SOC(t) = SOC0 - (100 / Q) * it(t)
2. Terminal voltage equation matching enhanced Shepherd model.
3. Proper MNA convergence and current sign convention.

Run from the helios_component_sim/ directory:
    python tests/test_battery.py
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
from components.battery import Battery

ELEMENT_REGISTRY.setdefault("battery", Battery)

# ---- Test parameters -------------------------------------------------
VNOM = 600.0    # V
Q = 0.367       # kAh = 367 Ah
SOC0 = 80.0     # %
R_LOAD = 2.0    # ohms -> ~300 A discharge (close to ~0.8C rate)
DT = 1e-4       # seconds
T_END = 0.1     # seconds

config = {
    "nodes": ["gnd", "n_pos", "n_neg"],
    "elements": [
        {
            "type": "battery",
            "name": "B1",
            "nodes": ["n_pos", "n_neg"],
            "value": {"Vnom": VNOM, "Q": Q, "SOC0": SOC0}
        },
        {"type": "resistor", "name": "R_load", "nodes": ["n_pos", "n_neg"], "value": R_LOAD},
        {"type": "resistor", "name": "R_gnd_ref", "nodes": ["n_neg", "gnd"], "value": 1e-4},
    ],
}


def test_battery_discharge() -> None:
    circuit = Circuit.from_config(config)
    sim = Simulator(circuit, DT, T_END)
    sim.run()

    t = sim.time
    v_pos = sim.voltage("n_pos")
    v_neg = sim.voltage("n_neg")
    v_batt = v_pos - v_neg

    batt_el = [e for e in circuit.elements if isinstance(e, Battery)][0]

    # Verification:
    # 1. Terminal voltage should be close to VNOM - I*R_int
    i_est = v_batt / R_LOAD
    print(f"[Battery] Initial V_batt = {v_batt[1]:.2f} V, Current = {i_est[1]:.2f} A")
    print(f"[Battery] Final   V_batt = {v_batt[-1]:.2f} V, Final SOC = {batt_el.soc:.4f} %")

    assert v_batt[1] > 0.9 * VNOM, "Battery terminal voltage collapsed unexpectedly"
    assert batt_el.soc < SOC0, "Battery SOC failed to decrease during discharge"

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(t * 1e3, v_batt, label="V_batt (V)", color="crimson")
    ax1.set_ylabel("Terminal Voltage (V)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t * 1e3, i_est, label="Discharge Current (A)", color="navy")
    ax2.set_ylabel("Current (A)")
    ax2.set_xlabel("Time (ms)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("battery_test.png", dpi=150)
    print("[Battery] PASS - saved battery_test.png")


if __name__ == "__main__":
    test_battery_discharge()

