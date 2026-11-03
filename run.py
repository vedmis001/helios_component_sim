"""
Entry point: load a circuit config, run the simulation, save results to CSV.

Usage (from the helios_sim/ directory):
    python run.py --config configs/rc_test.json
    python run.py --config configs/rc_test.json --dt 5e-7 --t-end 0.02 --out results/rc_fine.csv

Config format (JSON):
    {
      "dt": 1e-6,
      "t_end": 0.05,
      "nodes": ["gnd", "n1", "n2"],
      "elements": [
        {"type": "vsource",   "name": "Vs", "nodes": ["n1", "gnd"], "value": 1.0},
        {"type": "resistor",  "name": "R1", "nodes": ["n1", "n2"],  "value": 100.0},
        {"type": "capacitor", "name": "C1", "nodes": ["n2", "gnd"], "value": 1e-4}
      ]
    }

JSON can only hold constant source values. Time-varying sources (callables)
have to be built in Python, as in the tests.
"""
import argparse
import json
import os

import numpy as np

from core.circuit import Circuit, ELEMENT_REGISTRY
from core.simulator import Simulator
from components.transformer import Transformer

ELEMENT_REGISTRY["transformer"] = Transformer


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_csv(sim: Simulator, path: str) -> None:
    """Columns: time, v(<node>) for every non-ground node, i(<source>) for every voltage source."""
    circuit = sim.circuit
    nodes = sorted((idx, name) for name, idx in circuit.node_map.items() if idx >= 0)

    header = (
        ["time"]
        + [f"v({name})" for _, name in nodes]
        + [f"i({vs.name})" for vs in circuit.vsources]
    )
    columns = (
        [sim.time]
        + [sim.x[:, idx] for idx, _ in nodes]
        + [sim.x[:, vs.branch_idx] for vs in circuit.vsources]
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savetxt(path, np.column_stack(columns), delimiter=",", header=",".join(header), comments="")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a circuit simulation from a JSON config.")
    parser.add_argument("--config", required=True, help="path to JSON config")
    parser.add_argument("--dt", type=float, help="timestep in seconds (overrides config)")
    parser.add_argument("--t-end", type=float, help="end time in seconds (overrides config)")
    parser.add_argument("--out", help="output CSV path (default: results/<config name>.csv)")
    args = parser.parse_args()

    config = load_config(args.config)
    dt = args.dt if args.dt is not None else config["dt"]
    t_end = args.t_end if args.t_end is not None else config["t_end"]

    stem = os.path.splitext(os.path.basename(args.config))[0]
    out_path = args.out or os.path.join("results", f"{stem}.csv")

    circuit = Circuit.from_config(config)
    sim = Simulator(circuit, dt, t_end)
    sim.run()
    save_csv(sim, out_path)

    print(f"ran {sim.n_steps} steps, dt = {dt:g} s, t_end = {t_end:g} s")
    for name, idx in sorted(circuit.node_map.items(), key=lambda kv: kv[1]):
        if idx >= 0:
            print(f"  final v({name}) = {sim.voltage(name)[-1]:.6g} V")
    print(f"saved {out_path}")

    # TODO (later): add a --plot flag that saves a PNG of the node voltages


if __name__ == "__main__":
    main()
