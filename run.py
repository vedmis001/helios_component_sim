"""
Entry point: load a circuit config, run the simulation, save results to CSV.

Usage (from the helios_component_sim/ directory):
    python run.py --config configs/rc_test.json
    python run.py --config configs/ups_slice_test.json --mode claude --plot
    python run.py --config configs/rlc_test.json --mode compare --plot
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.circuit import Circuit, ELEMENT_REGISTRY
from core.simulator import Simulator
from core.adaptive import TimeStepSettings, ClaudeTimeStepAgent
from components.transformer import Transformer
from components.battery import Battery
from components.buckboost import BuckBoost
from components.machine import InductionMachine

ELEMENT_REGISTRY["transformer"] = Transformer
ELEMENT_REGISTRY["battery"] = Battery
ELEMENT_REGISTRY["buckboost"] = BuckBoost
ELEMENT_REGISTRY["machine"] = InductionMachine


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_csv(sim: Simulator, path: str) -> None:
    """Columns: time, v(<node>) for every non-ground node, i(<source>) for branch elements."""
    circuit = sim.circuit
    nodes = sorted((idx, name) for name, idx in circuit.node_map.items() if idx >= 0)

    all_branches = circuit.vsources + getattr(circuit, "branch_elements", [])

    header = (
        ["time"]
        + [f"v({name})" for _, name in nodes]
        + [f"i({el.name})" for el in all_branches]
    )
    columns = (
        [sim.time]
        + [sim.x[:, idx] for idx, _ in nodes]
        + [sim.x[:, el.branch_idx] for el in all_branches]
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savetxt(path, np.column_stack(columns), delimiter=",", header=",".join(header), comments="")


def save_plot(sim: Simulator, plot_path: str) -> None:
    """Save a PNG plot of node voltages and (if adaptive) step size / LTE."""
    circuit = sim.circuit
    nodes = sorted((idx, name) for name, idx in circuit.node_map.items() if idx >= 0)
    t_ms = sim.time * 1e3

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    for idx, name in nodes:
        ax1.plot(t_ms, sim.x[:, idx], label=f"v({name})", lw=1.4)
    ax1.set_ylabel("Node Voltage (V)")
    ax1.set_title(
        f"Mode: {sim.stats.get('mode', 'fixed').upper()} | "
        f"Accepted Steps: {sim.stats.get('n_accepted_steps', sim.n_steps)} | "
        f"Total Solves: {sim.stats.get('total_solves', sim.n_steps)}"
    )
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    if len(sim.dt_history) > 0:
        ax2.step(t_ms[1:], sim.dt_history * 1e6, where="post", color="darkgreen", label="dt (us)")
    ax2.set_ylabel("Step Size dt (us)")
    ax2.set_xlabel("Time (ms)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a circuit simulation from a JSON config.")
    parser.add_argument("--config", required=True, help="path to JSON config")
    parser.add_argument("--dt", type=float, help="timestep in seconds (overrides config)")
    parser.add_argument("--t-end", type=float, help="end time in seconds (overrides config)")
    parser.add_argument("--out", help="output CSV path (default: results/<config name>.csv)")
    parser.add_argument(
        "--mode",
        choices=["fixed", "pscad", "claude", "compare"],
        default="fixed",
        help="stepping mode: fixed (default), pscad (LTE + event window), claude (agentic), or compare",
    )
    parser.add_argument("--tol-lte", type=float, default=1e-2, help="LTE tolerance for adaptive stepping")
    parser.add_argument("--claude-api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--dense", action="store_true", help="use dense zero matrix instead of SparseMatrix")
    parser.add_argument("--plot", action="store_true", help="save a PNG plot of node voltages and step sizes")
    args = parser.parse_args()

    config = load_config(args.config)
    dt = args.dt if args.dt is not None else config["dt"]
    t_end = args.t_end if args.t_end is not None else config["t_end"]

    stem = os.path.splitext(os.path.basename(args.config))[0]
    out_path = args.out or os.path.join("results", f"{stem}.csv")

    use_sparse = not args.dense

    ev_times = np.asarray(config.get("event_times", [3.0 * dt, 0.5 * t_end]), dtype=float)

    if args.mode == "compare":
        print(f"=== Benchmarking {stem} (t_end = {t_end:g} s, base dt = {dt:g} s) ===")
        for m in ("fixed", "pscad", "claude"):
            c = Circuit.from_config(config, use_sparse=use_sparse)
            settings = TimeStepSettings(
                tol_lte=args.tol_lte,
                dt_min=dt * 0.2,
                dt_max=min(t_end / 15.0, dt * 40.0),
                dt_normal=dt * 2.0,
                dt_event=dt,
                t_event_window=min(t_end * 0.06, dt * 150.0),
                event_times=ev_times,
            )
            agent = ClaudeTimeStepAgent(api_key=args.claude_api_key) if m == "claude" else None
            s = Simulator(c, dt, t_end, mode=m, settings=settings, agent=agent)
            stats = s.run(m)
            print(
                f"  [{m.upper():6s}] solves={stats['total_solves']:5d} "
                f"(accepted={stats['n_accepted_steps']:5d}, rejected={stats['n_rejected_steps']:3d}) | "
                f"dt_mean={stats['dt_mean_used']:.2e}s | "
                f"density={stats['matrix_density']*100:.1f}% (nnz={stats['matrix_nnz']}, cap={stats['triplet_capacity']})"
            )
            if m == "claude":
                sim = s
    else:
        circuit = Circuit.from_config(config, use_sparse=use_sparse)
        settings = TimeStepSettings(
            tol_lte=args.tol_lte,
            dt_min=dt * 0.2,
            dt_max=min(t_end / 15.0, dt * 40.0),
            dt_normal=dt * 2.0,
            dt_event=dt,
            t_event_window=min(t_end * 0.06, dt * 150.0),
            event_times=ev_times,
        )
        agent = ClaudeTimeStepAgent(api_key=args.claude_api_key) if args.mode == "claude" else None
        sim = Simulator(circuit, dt, t_end, mode=args.mode, settings=settings, agent=agent)
        sim.run(args.mode)

    save_csv(sim, out_path)

    st = sim.stats
    print(
        f"ran {sim.n_steps} accepted steps (total solves = {st.get('total_solves', sim.n_steps)}, "
        f"rejected = {st.get('n_rejected_steps', 0)}), mode = {st.get('mode', args.mode)}, "
        f"matrix density = {st.get('matrix_density', 0.0)*100:.1f}% "
        f"(nnz = {st.get('matrix_nnz', 0)}, buffer capacity = {st.get('triplet_capacity', 0)})"
    )
    for name, idx in sorted(sim.circuit.node_map.items(), key=lambda kv: kv[1]):
        if idx >= 0:
            print(f"  final v({name}) = {sim.voltage(name)[-1]:.6g} V")
    print(f"saved {out_path}")

    if args.plot:
        plot_path = os.path.splitext(out_path)[0] + ".png"
        save_plot(sim, plot_path)
        print(f"saved plot {plot_path}")


if __name__ == "__main__":
    main()
