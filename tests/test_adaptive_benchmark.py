"""
Comprehensive Validation & Benchmark Suite for:
1. Vectorized COO Triplet (`rows`, `cols`, `vals`) Sparse Matrix (`SparseMatrix`)
   with capacity = `n_elements * 100` and density/sparsity tracking.
2. Adaptive Time-Stepping Comparison across three regimes:
   - Fixed Time-Stepping (Baseline)
   - PSCAD-Style Adaptive Time-Stepping (`get_next_time` + `calc_LTE_simple_trap` + `compute_adaptive_dt`)
   - Claude Agentic Adaptive Time-Stepping (`ClaudeTimeStepAgent` with proactive curvature lookahead
     and single-retry optimal LTE back-off)

Run from the helios_component_sim/ directory:
    python tests/test_adaptive_benchmark.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.circuit import Circuit, ELEMENT_REGISTRY, SingularMatrixError
from core.sparse import SparseMatrix
from core.simulator import Simulator
from core.adaptive import (
    TimeStepSettings,
    get_next_time,
    calc_LTE_simple_trap,
    compute_adaptive_dt,
    ClaudeTimeStepAgent,
)
from components.transformer import Transformer
from components.battery import Battery
from components.buckboost import BuckBoost
from components.machine import InductionMachine

ELEMENT_REGISTRY["transformer"] = Transformer
ELEMENT_REGISTRY["battery"] = Battery
ELEMENT_REGISTRY["buckboost"] = BuckBoost
ELEMENT_REGISTRY["machine"] = InductionMachine


def test_sparse_triplet_buffer_and_density() -> None:
    """Verify triplet vector allocation (n_elements * 100), density metrics, and dense equivalence."""
    print("=== 1. Vectorized SparseMatrix (row, col, val) & Density Verification ===")

    # 1a. Verify exact match between SparseMatrix and Dense on UPS slice
    with open("configs/ups_slice_test.json") as f:
        ups_cfg = json.load(f)

    c_sparse = Circuit.from_config(ups_cfg, use_sparse=True)
    c_dense = Circuit.from_config(ups_cfg, use_sparse=False)

    d_info = c_sparse.density_summary(1e-5, 1e-5)
    expected_cap = len(ups_cfg["elements"]) * 100
    assert d_info["allocated_capacity"] == expected_cap, (
        f"Expected capacity {expected_cap}, got {d_info['allocated_capacity']}"
    )

    sim_sparse = Simulator(c_sparse, 1e-4, 0.01, mode="fixed")
    sim_dense = Simulator(c_dense, 1e-4, 0.01, mode="fixed")
    sim_sparse.run()
    sim_dense.run()

    max_diff = float(np.max(np.abs(sim_sparse.x - sim_dense.x)))
    print(
        f"  [UPS Slice (6x6)]       nnz={d_info['nnz']:3d}/{d_info['total_entries']:4d} | "
        f"density={d_info['density']*100:5.1f}% | sparsity={d_info['sparsity']*100:5.1f}% | "
        f"triplet_cap={d_info['allocated_capacity']:4d} | max|x_sparse - x_dense| = {max_diff:.2e}"
    )
    assert max_diff < 1e-10, f"Sparse and dense solutions diverged: {max_diff}"

    # 1b. Verify density scaling on a 50-stage RC ladder network (51 nodes, 101 elements)
    n_stages = 50
    nodes = ["gnd"] + [f"n{k}" for k in range(1, n_stages + 2)]
    elements = [{"type": "vsource", "name": "Vs", "nodes": ["n1", "gnd"], "value": 10.0}]
    for k in range(1, n_stages + 1):
        elements.append({"type": "resistor", "name": f"R{k}", "nodes": [f"n{k}", f"n{k+1}"], "value": 10.0})
        elements.append({"type": "capacitor", "name": f"C{k}", "nodes": [f"n{k+1}", "gnd"], "value": 1e-5})

    ladder_cfg = {"nodes": nodes, "elements": elements}
    c_ladder = Circuit.from_config(ladder_cfg, use_sparse=True)
    lad_info = c_ladder.density_summary(1e-5, 1e-5)
    print(
        f"  [50-Stage Ladder (52x52)] nnz={lad_info['nnz']:3d}/{lad_info['total_entries']:4d} | "
        f"density={lad_info['density']*100:5.1f}% | sparsity={lad_info['sparsity']*100:5.1f}% | "
        f"triplet_cap={lad_info['allocated_capacity']:5d} (n_elements*100 = {len(elements)*100})"
    )
    assert lad_info["allocated_capacity"] == len(elements) * 100
    assert lad_info["density"] < 0.08, "50-stage ladder should be >92% sparse"
    print("  [SparseMatrix] PASS\n")


def build_switched_rlc_config():
    """
    Underdamped RLC circuit driven by a multi-event pulsed voltage source:
      - Event 1 at t = 0.0 ms:   step from 0 V -> 10 V
      - Event 2 at t = 8.0 ms:   step from 10 V -> -5 V
      - Event 3 at t = 16.0 ms:  step from -5 V -> 15 V
    """
    def v_pulsed(t: float) -> float:
        if t <= 8e-3:
            return 10.0
        elif t <= 16e-3:
            return -5.0
        return 15.0

    config = {
        "nodes": ["gnd", "n1", "n2", "n3"],
        "elements": [
            {"type": "vsource", "name": "Vs", "nodes": ["n1", "gnd"], "value": v_pulsed},
            {"type": "resistor", "name": "R1", "nodes": ["n1", "n2"], "value": 12.0},
            {"type": "inductor", "name": "L1", "nodes": ["n2", "n3"], "value": 4e-3},
            {"type": "capacitor", "name": "C1", "nodes": ["n3", "gnd"], "value": 25e-6},
        ],
    }
    event_times = np.array([8e-3, 16e-3])
    return config, event_times


def benchmark_switched_rlc() -> dict:
    """
    Compare Fixed vs. PSCAD Adaptive vs. Claude Agentic Adaptive on the
    multi-event switched RLC transient (t_end = 25 ms).
    """
    print("=== 2. Scenario A: Multi-Event Switched RLC Transient (t_end = 25 ms) ===")
    cfg, event_times = build_switched_rlc_config()
    dt_base = 5e-6
    t_end = 25e-3

    # 1. Fixed time step baseline
    sim_fixed = Simulator(Circuit.from_config(cfg), dt_base, t_end, mode="fixed")
    st_fixed = sim_fixed.run("fixed")

    # Shared LTE settings for PSCAD and Claude modes
    settings = TimeStepSettings(
        tol_lte=2.5e-3,
        dt_min=2e-6,
        dt_max=4e-4,
        dt_normal=4e-5,
        dt_event=5e-6,
        t_event_window=8e-4,
        beta=0.8,
        event_times=event_times,
    )

    # 2. PSCAD-style adaptive (event state machine + classical LTE step & back-off)
    sim_pscad = Simulator(Circuit.from_config(cfg), dt_base, t_end, mode="pscad", settings=settings)
    st_pscad = sim_pscad.run_adaptive(mode="pscad", settings=settings, event_times=event_times)

    # 3. Claude Agentic adaptive (trajectory/curvature lookahead + single-retry optimal back-off)
    agent = ClaudeTimeStepAgent()
    sim_claude = Simulator(Circuit.from_config(cfg), dt_base, t_end, mode="claude", settings=settings, agent=agent)
    st_claude = sim_claude.run_adaptive(mode="claude", settings=settings, event_times=event_times, agent=agent)

    # Compute relative RMSE of capacitor voltage v(n3) against the fine fixed-step trajectory
    t_ref = sim_fixed.time
    v_ref = sim_fixed.voltage("n3")
    norm = float(np.max(np.abs(v_ref)))

    v_pscad_interp = np.interp(t_ref, sim_pscad.time, sim_pscad.voltage("n3"))
    v_claude_interp = np.interp(t_ref, sim_claude.time, sim_claude.voltage("n3"))

    rmse_pscad = float(np.sqrt(np.mean((v_pscad_interp - v_ref) ** 2)) / norm)
    rmse_claude = float(np.sqrt(np.mean((v_claude_interp - v_ref) ** 2)) / norm)

    print(f"  {'Mode':<16s} | {'Accepted':>8s} | {'Rejected':>8s} | {'Total Solves':>12s} | {'Solve Savings':>13s} | {'Rel RMSE':>10s}")
    print("  " + "-" * 79)
    print(
        f"  {'1. Fixed Step':<16s} | {st_fixed['n_accepted_steps']:8d} | {st_fixed['n_rejected_steps']:8d} | "
        f"{st_fixed['total_solves']:12d} | {'1.00x (base)':>13s} | {'0.00e+00':>10s}"
    )
    print(
        f"  {'2. PSCAD Adapt':<16s} | {st_pscad['n_accepted_steps']:8d} | {st_pscad['n_rejected_steps']:8d} | "
        f"{st_pscad['total_solves']:12d} | {st_fixed['total_solves']/st_pscad['total_solves']:12.2f}x | {rmse_pscad:10.2e}"
    )
    print(
        f"  {'3. Claude Agent':<16s} | {st_claude['n_accepted_steps']:8d} | {st_claude['n_rejected_steps']:8d} | "
        f"{st_claude['total_solves']:12d} | {st_fixed['total_solves']/st_claude['total_solves']:12.2f}x | {rmse_claude:10.2e}\n"
    )

    assert st_pscad["total_solves"] < st_fixed["total_solves"], "PSCAD adaptive should use fewer solves than fixed"
    assert st_claude["total_solves"] < st_pscad["total_solves"], "Claude agentic should use fewer solves than PSCAD adaptive"
    assert st_claude["n_rejected_steps"] <= st_pscad["n_rejected_steps"], "Claude agentic should have fewer or equal rejected steps"
    assert rmse_pscad < 2e-2 and rmse_claude < 2e-2, "Adaptive trajectories must remain accurate"

    # Save 3-panel comparison plot
    os.makedirs("results", exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax1.plot(sim_fixed.time * 1e3, v_ref, color="black", lw=1.6, label=f"1. Fixed Step ({st_fixed['total_solves']} solves)")
    ax1.plot(sim_pscad.time * 1e3, sim_pscad.voltage("n3"), "--", color="royalblue", lw=1.4,
             label=f"2. PSCAD Adaptive ({st_pscad['total_solves']} solves, {st_pscad['n_rejected_steps']} rej)")
    ax1.plot(sim_claude.time * 1e3, sim_claude.voltage("n3"), "o-", color="crimson", lw=1.2, markersize=2.5,
             label=f"3. Claude Agentic ({st_claude['total_solves']} solves, {st_claude['n_rejected_steps']} rej)")
    for ev in event_times:
        ax1.axvline(ev * 1e3, color="gray", ls=":", alpha=0.7)
    ax1.set_ylabel("Capacitor Voltage v(n3) [V]")
    ax1.set_title("Multi-Event Switched RLC: Fixed vs. PSCAD Adaptive vs. Claude Agentic Stepping")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.axhline(dt_base * 1e6, color="black", ls=":", lw=1.2, label="Fixed dt (5 us)")
    ax2.step(sim_pscad.time[1:] * 1e3, sim_pscad.dt_history * 1e6, where="post", color="royalblue", lw=1.3, label="PSCAD dt (us)")
    ax2.step(sim_claude.time[1:] * 1e3, sim_claude.dt_history * 1e6, where="post", color="crimson", lw=1.4, label="Claude Agentic dt (us)")
    ax2.set_yscale("log")
    ax2.set_ylabel("Step Size dt [us]")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    ax3.plot(sim_fixed.time[1:] * 1e3, np.arange(1, len(sim_fixed.time)), color="black", lw=1.5, label="Fixed Cumulative Solves")
    ax3.plot(sim_pscad.time[1:] * 1e3, np.linspace(1, st_pscad["total_solves"], len(sim_pscad.time) - 1),
             color="royalblue", lw=1.5, label="PSCAD Cumulative Solves")
    ax3.plot(sim_claude.time[1:] * 1e3, np.linspace(1, st_claude["total_solves"], len(sim_claude.time) - 1),
             color="crimson", lw=1.8, label="Claude Agentic Cumulative Solves")
    ax3.set_ylabel("Cumulative MNA/NR Solves")
    ax3.set_xlabel("Time [ms]")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/adaptive_benchmark_rlc.png", dpi=150)
    plt.close(fig)
    print("  Saved plot: results/adaptive_benchmark_rlc.png\n")

    return {"fixed": st_fixed, "pscad": st_pscad, "claude": st_claude}


def benchmark_ups_slice() -> dict:
    """
    Compare Fixed vs. PSCAD Adaptive vs. Claude Agentic Adaptive on the
    HELIOS UPS Slice (Battery + Bidirectional Buck-Boost Converter + DC Link).
    """
    print("=== 3. Scenario B: HELIOS UPS Slice (configs/ups_slice_test.json, t_end = 50 ms) ===")
    with open("configs/ups_slice_test.json") as f:
        ups_cfg = json.load(f)

    dt_base = ups_cfg["dt"]      # 10 us
    t_end = ups_cfg["t_end"]     # 50 ms

    # 1. Fixed Step
    sim_fixed = Simulator(Circuit.from_config(ups_cfg), dt_base, t_end, mode="fixed")
    st_fixed = sim_fixed.run("fixed")

    settings = TimeStepSettings(
        tol_lte=5e-2,
        dt_min=5e-6,
        dt_max=5e-4,
        dt_normal=4e-5,
        dt_event=1e-5,
        t_event_window=2e-3,
        beta=0.8,
        event_times=np.array([3e-5, 25e-3]),
    )

    # 2. PSCAD Adaptive
    sim_pscad = Simulator(Circuit.from_config(ups_cfg), dt_base, t_end, mode="pscad", settings=settings)
    st_pscad = sim_pscad.run_adaptive(mode="pscad", settings=settings)

    # 3. Claude Agentic Adaptive
    agent = ClaudeTimeStepAgent()
    sim_claude = Simulator(Circuit.from_config(ups_cfg), dt_base, t_end, mode="claude", settings=settings, agent=agent)
    st_claude = sim_claude.run_adaptive(mode="claude", settings=settings, agent=agent)

    t_ref = sim_fixed.time
    vdc_ref = sim_fixed.voltage("vdc_p") - sim_fixed.voltage("vdc_n")
    norm = float(np.max(np.abs(vdc_ref)))

    vdc_pscad = sim_pscad.voltage("vdc_p") - sim_pscad.voltage("vdc_n")
    vdc_claude = sim_claude.voltage("vdc_p") - sim_claude.voltage("vdc_n")

    rmse_pscad = float(np.sqrt(np.mean((np.interp(t_ref, sim_pscad.time, vdc_pscad) - vdc_ref) ** 2)) / norm)
    rmse_claude = float(np.sqrt(np.mean((np.interp(t_ref, sim_claude.time, vdc_claude) - vdc_ref) ** 2)) / norm)

    print(f"  {'Mode':<16s} | {'Accepted':>8s} | {'Rejected':>8s} | {'Total Solves':>12s} | {'Solve Savings':>13s} | {'Rel RMSE':>10s}")
    print("  " + "-" * 79)
    print(
        f"  {'1. Fixed Step':<16s} | {st_fixed['n_accepted_steps']:8d} | {st_fixed['n_rejected_steps']:8d} | "
        f"{st_fixed['total_solves']:12d} | {'1.00x (base)':>13s} | {'0.00e+00':>10s}"
    )
    print(
        f"  {'2. PSCAD Adapt':<16s} | {st_pscad['n_accepted_steps']:8d} | {st_pscad['n_rejected_steps']:8d} | "
        f"{st_pscad['total_solves']:12d} | {st_fixed['total_solves']/st_pscad['total_solves']:12.2f}x | {rmse_pscad:10.2e}"
    )
    print(
        f"  {'3. Claude Agent':<16s} | {st_claude['n_accepted_steps']:8d} | {st_claude['n_rejected_steps']:8d} | "
        f"{st_claude['total_solves']:12d} | {st_fixed['total_solves']/st_claude['total_solves']:12.2f}x | {rmse_claude:10.2e}\n"
    )

    assert st_pscad["total_solves"] < st_fixed["total_solves"]
    assert st_claude["total_solves"] < st_pscad["total_solves"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax1.plot(sim_fixed.time * 1e3, vdc_ref, color="black", lw=1.6, label=f"1. Fixed Step ({st_fixed['total_solves']} solves)")
    ax1.plot(sim_pscad.time * 1e3, vdc_pscad, "--", color="royalblue", lw=1.4,
             label=f"2. PSCAD Adaptive ({st_pscad['total_solves']} solves, {st_pscad['n_rejected_steps']} rej)")
    ax1.plot(sim_claude.time * 1e3, vdc_claude, "o-", color="crimson", lw=1.2, markersize=2.5,
             label=f"3. Claude Agentic ({st_claude['total_solves']} solves, {st_claude['n_rejected_steps']} rej)")
    ax1.set_ylabel("DC Link Voltage V_dc [V]")
    ax1.set_title("HELIOS UPS Slice: Fixed vs. PSCAD Adaptive vs. Claude Agentic Stepping")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.axhline(dt_base * 1e6, color="black", ls=":", lw=1.2, label="Fixed dt (10 us)")
    ax2.step(sim_pscad.time[1:] * 1e3, sim_pscad.dt_history * 1e6, where="post", color="royalblue", lw=1.3, label="PSCAD dt (us)")
    ax2.step(sim_claude.time[1:] * 1e3, sim_claude.dt_history * 1e6, where="post", color="crimson", lw=1.4, label="Claude Agentic dt (us)")
    ax2.set_yscale("log")
    ax2.set_ylabel("Step Size dt [us]")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    ax3.plot(sim_fixed.time[1:] * 1e3, np.arange(1, len(sim_fixed.time)), color="black", lw=1.5, label="Fixed Cumulative Solves")
    ax3.plot(sim_pscad.time[1:] * 1e3, np.linspace(1, st_pscad["total_solves"], len(sim_pscad.time) - 1),
             color="royalblue", lw=1.5, label="PSCAD Cumulative Solves")
    ax3.plot(sim_claude.time[1:] * 1e3, np.linspace(1, st_claude["total_solves"], len(sim_claude.time) - 1),
             color="crimson", lw=1.8, label="Claude Agentic Cumulative Solves")
    ax3.set_ylabel("Cumulative MNA/NR Solves")
    ax3.set_xlabel("Time [ms]")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/adaptive_benchmark_ups.png", dpi=150)
    plt.close(fig)
    print("  Saved plot: results/adaptive_benchmark_ups.png")

    return {"fixed": st_fixed, "pscad": st_pscad, "claude": st_claude}


if __name__ == "__main__":
    test_sparse_triplet_buffer_and_density()
    benchmark_switched_rlc()
    benchmark_ups_slice()
    print("\nALL SPARSE & ADAPTIVE BENCHMARK TESTS PASSED.")
